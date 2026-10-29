#!/usr/bin/env python3
"""Curate cached read speech without admitting unverified translated captions."""
import argparse
from collections import Counter
import fcntl
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import tarfile

from common import atomic_json, load_manifests, read_jsonl, read_tsv, sha256, text_key, validate_splits
from prefetch import SOURCES


def source_holdout(speaker):
    return int(hashlib.sha256(speaker.encode()).hexdigest()[:8], 16) % 10 == 0


def aishell_training_member(name):
    parts = Path(name).parts
    return len(parts) == 3 and parts[0] == "train" and parts[-1].endswith(".wav")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--core", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protected-test", type=Path, required=True)
    p.add_argument("--aishell-hours", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=20261029)
    a = p.parse_args()
    if a.aishell_hours <= 0:
        p.error("positive Mandarin duration required")
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (a.output / "data.json").exists():
            load_manifests(a.output, a.protected_test)
            return
        build(a)


def build(a):
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    core_train, dev, _ = load_manifests(a.core, a.protected_test)
    protected = read_jsonl(a.protected_test)
    catalog = json.loads((a.root / "prefetch.json").read_text())
    if catalog["status"] != "complete":
        raise ValueError("source intake is incomplete")
    completed = catalog["completed"]
    used = {}
    rejected = Counter()
    seconds = Counter()
    rows = []
    blocked_text = {text_key(r["text"]) for r in dev} | {text_key(r["reference"]) for r in protected}
    seen_hash = {r["sha256"] for r in dev + protected}
    dev_speakers = {r["speaker"] for r in dev if r.get("speaker")}

    def disk_check():
        if shutil.disk_usage(a.output).free < 100 * (1 << 30):
            raise RuntimeError("less than 100 GiB free; intake stopped")

    def cached(key):
        disk_check()
        item = completed[key]
        source = SOURCES[key.split("/", 1)[0]]
        if (item["source"], item["revision"], item["license"]) != source:
            raise ValueError(f"changed pinned source: {key}")
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"changed cached file: {key}")
        used[key] = item
        return path

    def progress():
        value = {"clips": len(rows), "hours": {k: v / 3600 for k, v in seconds.items()},
                 "rejected": dict(rejected)}
        atomic_json(a.output / "progress.json", value)
        print(json.dumps(value), flush=True)

    def append(row):
        rows.append(row)
        seen_hash.add(row["sha256"])
        seconds[row["source"] + "/" + row["language"]] += row["seconds"]
        if len(rows) % 1000 == 0:
            disk_check()
            progress()

    def keep(source_name, lang, identifier, speaker, text, audio, rate):
        text = " ".join(text.split())
        key = text_key(text)
        if not key or key in blocked_text or speaker in dev_speakers:
            rejected["heldout_text_or_speaker"] += 1
            return
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != 16000:
            factor = gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // factor, rate // factor)
        duration = len(audio) / 16000
        if not 1 <= duration <= 15 or not np.isfinite(audio).all():
            rejected["duration_or_nonfinite"] += 1
            return
        source = SOURCES[source_name]
        sample_id = f"{source_name}/{identifier}"
        file_id = hashlib.sha256(sample_id.encode()).hexdigest()
        path = a.output / "wav" / f"expanded-{file_id}.wav"
        sf.write(path, audio, 16000, subtype="PCM_16")
        digest = sha256(path)
        if digest in seen_hash:
            path.unlink()
            rejected["audio_duplicate_or_sealed"] += 1
            return
        append({"id": sample_id, "language": lang, "speaker": speaker,
                "audio": str(path.relative_to(a.output)), "text": text,
                "seconds": duration, "sha256": digest,
                "source": source[0], "source_revision": source[1], "license": source[2]})

    disk_check()
    (a.output / "wav").mkdir(exist_ok=True)
    # Keep the complete first clean mixture, including English replay. All
    # additions pass the same held-out text/audio exclusions as these rows.
    # Repeated training sentences from distinct recordings are useful acoustic
    # diversity; deduplicate audio, while keeping all held-out text excluded.
    for row in core_train:
        source = a.core / row["audio"]
        target = a.output / row["audio"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        append(row)

    zeroth_files = sorted(k for k in completed if k.startswith("zeroth/data/train-") and k.endswith(".parquet"))
    random.Random(a.seed).shuffle(zeroth_files)
    for key in zeroth_files:
        for batch in pq.ParquetFile(cached(key)).iter_batches(batch_size=32):
            for item in batch.to_pylist():
                speaker = f"zeroth/{item['speaker_id']}"
                if source_holdout(speaker):
                    rejected["zeroth_source_dev_speaker"] += 1
                    continue
                if text_key(item["text"]) in blocked_text:
                    rejected["heldout_text_or_speaker"] += 1
                    continue
                audio, rate = sf.read(io.BytesIO(item["audio"]["bytes"]), dtype="float32")
                keep("zeroth", "ko", str(item["id"]), speaker, item["text"], audio, rate)
        progress()

    # Expand Japanese only from validated CV train. Reserve ALL official dev
    # speakers/sentences, including those absent from the small pilot dev.
    metadata = {}
    for split in ("train", "dev"):
        metadata[split] = read_tsv(cached(f"cv17/transcript/ja/{split}.tsv"))
    held_speakers = {r["client_id"] for r in metadata["dev"]}
    held_text = {text_key(r["sentence"]) for r in metadata["dev"]}
    wanted = {Path(r["path"]).name: r for r in metadata["train"]
              if r["client_id"] not in held_speakers and text_key(r["sentence"]) not in held_text}
    for key in sorted(k for k in completed if k.startswith("cv17/audio/ja/train/") and k.endswith(".tar")):
        with tarfile.open(cached(key)) as archive:
            for member in archive:
                item = wanted.get(Path(member.name).name)
                if not member.isfile() or item is None:
                    continue
                if text_key(item["sentence"]) in blocked_text:
                    rejected["heldout_text_or_speaker"] += 1
                    continue
                with archive.extractfile(member) as stream:
                    audio, rate = sf.read(io.BytesIO(stream.read()), dtype="float32")
                keep("cv17", "ja", Path(item["path"]).stem, f"cv17/{item['client_id']}",
                     item["sentence"], audio, rate)
        progress()

    transcript = cached("aishell/data_aishell/transcript/aishell_transcript_v0.8.txt")
    # The official Mandarin text is character/word segmented. Strip its
    # annotation spaces for natural transcript targets; CER is space-free.
    texts = {parts[0]: "".join(parts[1].split()) for line in transcript.read_text().splitlines()
             if len(parts := line.split(maxsplit=1)) == 2}
    aishell_files = sorted(k for k in completed if k.startswith("aishell/data_aishell/wav/") and k.endswith(".tar.gz"))
    random.Random(a.seed).shuffle(aishell_files)
    for key in aishell_files:
        if seconds[SOURCES["aishell"][0] + "/zh"] >= a.aishell_hours * 3600:
            break
        with tarfile.open(cached(key)) as archive:
            for member in archive:
                if not member.isfile() or not aishell_training_member(member.name):
                    continue
                identifier = Path(member.name).stem
                text = texts.get(identifier)
                if not text or text_key(text) in blocked_text:
                    rejected["missing_or_heldout_aishell_text"] += 1
                    continue
                with archive.extractfile(member) as stream:
                    audio, rate = sf.read(io.BytesIO(stream.read()), dtype="float32")
                keep("aishell", "zh", identifier, f"aishell/{Path(member.name).parts[1]}", text, audio, rate)
                if seconds[SOURCES["aishell"][0] + "/zh"] >= a.aishell_hours * 3600:
                    break
        progress()

    validate_splits(rows, dev, protected)
    random.Random(a.seed).shuffle(rows)
    (a.output / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    for row in dev:
        target = a.output / row["audio"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.core / row["audio"], target)
    shutil.copy2(a.core / "dev.jsonl", a.output / "dev.jsonl")
    for row in rows + dev:
        if sha256(a.output / row["audio"]) != row["sha256"]:
            raise ValueError(f"changed prepared audio: {row['id']}")
    atomic_json(a.output / "data.json", {
        "train_sha256": sha256(a.output / "train.jsonl"), "dev_sha256": sha256(a.output / "dev.jsonl"),
        "protected_manifest_sha256": sha256(a.protected_test), "core_data_sha256": sha256(a.core / "data.json"),
        "selection": {"recipe": "expanded-clean-v2", "seed": a.seed, "aishell_hours_cap": a.aishell_hours},
        "train_clips": len(rows), "dev_clips": len(dev), "train_hours": sum(seconds.values()) / 3600,
        "seconds_by_source_language": dict(seconds), "rejected": dict(rejected), "downloads": used,
        "preparation_code_sha256": {p.name: sha256(p) for p in
                                    (Path(__file__), Path(__file__).with_name("common.py"),
                                     Path(__file__).with_name("prefetch.py"))},
        "quarantined": {"espnet/yodas": "Caption inspection found translated lecture material; audio-language/alignment verification is required before use."},
        "sampling": "all usable cached Zeroth/CV17 Japanese train, seeded AISHELL speaker archive order; source holdouts retained",
    })
    progress()
    print(json.dumps({"ready": str(a.output)}), flush=True)


if __name__ == "__main__":
    main()
