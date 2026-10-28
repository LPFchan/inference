"""CPU-only data, reward and checkpoint-contract helpers."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("stt_score", ROOT / "scripts/stt-score.py")
_score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_score)
normalize = _score.normalize
LANGUAGES = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
        return digest.hexdigest()


def text_key(text):
    return normalize(text).replace(" ", "")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def reward(reference, hypothesis, language, korean_wer_weight=0.0):
    """CER for CJK, WER for English; no learned reward model or teacher calls."""
    import jiwer
    ref, hyp = normalize(reference), normalize(hypothesis)
    if language == "en":
        error = jiwer.wer(ref, hyp)
    else:
        error = jiwer.cer(ref.replace(" ", ""), hyp.replace(" ", ""))
    if language == "ko" and korean_wer_weight:
        error = (1 - korean_wer_weight) * error + korean_wer_weight * jiwer.wer(ref, hyp)
    return max(-1.0, 1.0 - error)


def advantages(rewards):
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    return [(value - mean) / (math.sqrt(variance) + 1e-6) for value in rewards]


def training_targets(prompt, completion):
    """Next-token targets include the first transcript token and terminal EOS."""
    if not prompt or not completion:
        raise ValueError("prompt and completion must be nonempty")
    ids = prompt + completion
    labels = ids[1:]
    mask = [False] * (len(prompt) - 1) + [True] * len(completion)
    return ids[:-1], labels, mask


def validate_splits(train, dev, protected):
    """Reject shared utterances, normalized text and speakers before training."""
    def keys(rows, field):
        return {row[field] for row in rows if row.get(field)}
    for rows in (train, dev):
        if len(keys(rows, "id")) != len(rows):
            raise ValueError("duplicate or missing sample id")
        if any(row["language"] not in LANGUAGES for row in rows):
            raise ValueError("unsupported language")
        if any(not text_key(row["text"]) for row in rows):
            raise ValueError("empty reference")
    for field in ("id", "sha256", "speaker"):
        if keys(train, field) & keys(dev, field):
            raise ValueError(f"train/dev {field} overlap")
    if {text_key(r["text"]) for r in train} & {text_key(r["text"]) for r in dev}:
        raise ValueError("train/dev text overlap")
    protected_text = {text_key(r["reference"]) for r in protected}
    for rows in (train, dev):
        if keys(rows, "sha256") & keys(protected, "sha256"):
            raise ValueError("frozen test audio overlap")
        if {text_key(r["text"]) for r in rows} & protected_text:
            raise ValueError("frozen test text overlap")


def load_manifests(directory, frozen):
    directory = Path(directory)
    metadata = json.loads((directory / "data.json").read_text())
    for name in ("train", "dev"):
        if sha256(directory / f"{name}.jsonl") != metadata[f"{name}_sha256"]:
            raise ValueError(f"changed {name} manifest")
    if sha256(frozen) != metadata["protected_manifest_sha256"]:
        raise ValueError("changed protected test manifest")
    train, dev = [read_jsonl(directory / f"{name}.jsonl") for name in ("train", "dev")]
    validate_splits(train, dev, read_jsonl(frozen))
    for row in train + dev:
        if sha256(directory / row["audio"]) != row["sha256"]:
            raise ValueError(f"changed audio: {row['id']}")
    return train, dev, metadata
