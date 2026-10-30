"""CPU-only plan, journal and summary for train.py's TRAIN-only greedy-versus-sampled probe."""
import hashlib
import json
from pathlib import Path

from common import sentence_epoch_order, sentence_key

ORDER = ("ko", "ja", "zh", "en")
SENTENCES = 200  # unique source sentences per language
SAMPLES = 8  # temperature draws per clip, after the greedy draw 0
SEED = 20261030
TEMPERATURE = 0.7
KOREAN_WER_WEIGHT = 0.5
SCOPE = ("TRAIN-only diagnostic of one adapter. Best-of-N ranks samples by the TRAIN reference: "
         "an optimistic oracle, not deployable decoding, held-out accuracy or an RL ceiling.")


def recipe():
    return {"sentences": SENTENCES, "samples": SAMPLES, "seed": SEED, "temperature": TEMPERATURE,
            "korean_wer_weight": KOREAN_WER_WEIGHT, "language_order": list(ORDER)}


def text_sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def plan(pools):
    """The first SENTENCES of each language's seeded sentence epoch, interleaved ko/ja/zh/en.

    A limit of N clips per language is the first 4N rows, so a smoke run is a
    prefix of the full plan. Rows carry hashes, never reference text.
    """
    order = sentence_epoch_order(pools, SEED)
    short = [lang for lang in ORDER if len(order[lang]) < SENTENCES]
    if short:
        raise ValueError(f"fewer than {SENTENCES} TRAIN sentences: {short}")
    return [{"index": k * len(ORDER) + position, "id": row["id"], "language": lang,
             "sentence": list(sentence_key(row)), "audio": row["audio"], "audio_sha256": row["sha256"],
             "reference_sha256": text_sha256(row["text"])}
            for k in range(SENTENCES) for position, lang in enumerate(ORDER)
            for row in [order[lang][k]]]


def plan_bytes(rows):
    return (json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=1) + "\n").encode()


def freeze_plan(path, rows):
    """Write the plan once; afterwards it must stay byte-identical."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != plan_bytes(rows):
            raise ValueError("changed probe plan")
        return
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(plan_bytes(rows))
    temporary.replace(path)


def draw_seed(clip_id, draw):
    """Stable per-clip/per-draw MLX seed, independent of order, resume or Python's hash."""
    digest = hashlib.sha256(f"train-probe:{SEED}:{clip_id}:{draw}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def expected(rows):
    """Immutable metadata of every decode in the full plan, keyed by (clip ID, draw)."""
    return {(row["id"], draw): {"clip": row["id"], "draw": draw, "index": row["index"],
                                "language": row["language"], "reference_sha256": row["reference_sha256"],
                                "temperature": 0.0 if draw == 0 else TEMPERATURE,
                                "seed": draw_seed(row["id"], draw)}
            for row in rows for draw in range(SAMPLES + 1)}


def _digest(record):
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def record(spec, contract_sha256, tokens, text, ended, seconds, reward):
    row = {**spec, "tokens": tokens, "text": text, "ended": ended, "seconds": seconds,
           "reward": reward, "contract_sha256": contract_sha256}
    row["record_sha256"] = _digest(row)
    return row


def saved_decodes(path, plan_decodes, contract_sha256):
    """Completed decodes; reject anything edited, duplicated, foreign or from another contract.

    Only a torn final line from an interrupted write is dropped.
    """
    path = Path(path)
    data = path.read_bytes() if path.exists() else b""
    lines = data.splitlines()
    repair = bool(data and not data.endswith(b"\n"))
    if repair:
        try:
            json.loads(lines[-1])
        except (json.JSONDecodeError, UnicodeDecodeError):
            lines.pop()
    saved = {}
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError(f"corrupt probe decode at line {number}") from None
        key = (row.get("clip"), row.get("draw")) if isinstance(row, dict) else None
        spec = plan_decodes.get(key)
        if (key in saved or spec is None or row.get("contract_sha256") != contract_sha256
                or any(row.get(name) != value for name, value in spec.items())
                or row.get("record_sha256") != _digest(row)):
            raise ValueError(f"saved probe decode drift at line {number}: {key}")
        saved[key] = row
    if repair:
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_bytes(b"".join(line + b"\n" for line in lines))
        temporary.replace(path)
    return saved


def _mean(values):
    return sum(values) / len(values)


def summary(rows, saved, parent_draws):
    """Per-language greedy/sampled/oracle rewards over the full plan; languages never pooled."""
    languages = {}
    for lang in ORDER:
        clips = [row for row in rows if row["language"] == lang]
        greedy = [saved[(row["id"], 0)] for row in clips]
        samples = [[saved[(row["id"], draw)] for draw in range(1, SAMPLES + 1)] for row in clips]
        best = [max(d["reward"] for d in group) for group in samples]
        errors = [(g["reward"], b) for g, b in zip(greedy, best) if g["reward"] < 1]
        sampled = [d for group in samples for d in group]
        languages[lang] = {
            "clips": len(clips), "unique_clips": len({row["id"] for row in clips}),
            "unique_sentences": len({tuple(row["sentence"]) for row in clips}),
            "greedy_reward": _mean([d["reward"] for d in greedy]),
            "sampled_reward": _mean([d["reward"] for d in sampled]),
            f"oracle_best_of_{SAMPLES}_reward": _mean(best),
            "oracle_headroom": _mean([max(0.0, b - g["reward"]) for g, b in zip(greedy, best)]),
            "greedy_errors": len(errors),
            "greedy_errors_with_better_sample": sum(b > g for g, b in errors),
            "greedy_errors_with_perfect_sample": sum(b == 1 for _, b in errors),
            "greedy_empty": sum(not d["text"].strip() for d in greedy),
            "greedy_truncated": sum(not d["ended"] for d in greedy),
            "sampled_empty": sum(not d["text"].strip() for d in sampled),
            "sampled_truncated": sum(not d["ended"] for d in sampled)}
        if parent_draws is not None:
            ids = {row["id"] for row in parent_draws}
            sentences = {(row["language"], sentence_key(row)) for row in parent_draws}
            languages[lang]["parent_drawn_recordings"] = sum(row["id"] in ids for row in clips)
            languages[lang]["parent_drawn_sentences"] = sum(
                (lang, tuple(row["sentence"])) in sentences for row in clips)
    return {"scope": SCOPE, "recipe": recipe(), "decodes": len(saved),
            "reward": "frozen reward() with KO WER weight 0.5; a completion without EOS scores -1 as in GRPO",
            "oracle_headroom": "mean over clips of max(0, best sampled reward - greedy reward)",
            "parent_exposure": None if parent_draws is None else (
                "plan clips among the adapter's own SFT draws, recomputed from its checkpoint's recorded "
                "sampler, seed, step and accumulate; earlier parents of that run are not included"),
            "decode_seconds": sum(d["seconds"] for d in saved.values()),
            "languages": languages}


def progress(limit, saved):
    """Overwritable smoke/progress counts; never a diagnostic result."""
    return {"scope": f"PARTIAL: stop target {limit} of {SENTENCES} clips per language; "
                     "plumbing counts only, not a diagnostic result",
            "clips_per_language": limit, "decodes": len(saved),
            "full_decodes": len(ORDER) * SENTENCES * (SAMPLES + 1),
            "empty": sum(not d["text"].strip() for d in saved.values()),
            "truncated": sum(not d["ended"] for d in saved.values()),
            "decode_seconds": sum(d["seconds"] for d in saved.values())}
