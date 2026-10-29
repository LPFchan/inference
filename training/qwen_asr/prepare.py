#!/usr/bin/env python3
"""Prepare pinned real speech for adaptation; frozen FLEURS is exclusion-only."""
import argparse
import fcntl
import io
import json
import random
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download
from scipy.signal import resample_poly

from common import atomic_json, load_manifests, read_jsonl, read_tsv, sha256, text_key, validate_splits

CV = ("fsicoli/common_voice_17_0", "8262c16bf297c87a9cd88c51997c4758ed7a8ba2", "cc0-1.0")
KO = ("kresnik/zeroth_korean", "1fe937899f828af822293d05e086200946088bdf", "cc-by-4.0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--protected-test", required=True, type=Path)
    parser.add_argument("--hours", type=float, default=0.5)
    parser.add_argument("--dev-hours", type=float, default=0.05)
    parser.add_argument("--english-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    if min(args.hours, args.dev_hours, args.english_ratio) <= 0:
        parser.error("positive duration and replay ratio required")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "data.json").exists():
            _, _, metadata = load_manifests(args.output, args.protected_test)
            expected = {"hours": args.hours, "dev_hours": args.dev_hours,
                        "english_ratio": args.english_ratio, "seed": args.seed}
            if metadata["selection"] != expected:
                raise ValueError("existing data has different settings; use a new output")
            print("Prepared data already complete and verified", flush=True)
            return
        prepare(args)


def prepare(args):
    protected = read_jsonl(args.protected_test)
    blocked_text = {text_key(row["reference"]) for row in protected}
    blocked_hash = {row["sha256"] for row in protected}
    seen_text, seen_hash, dev_speakers = set(), set(), set()
    rows = {"train": [], "dev": []}
    seconds = Counter()
    rejected = Counter()
    downloads = {}
    cache = args.output / "download-cache"
    wav_dir = args.output / "wav"
    wav_dir.mkdir(exist_ok=True)

    def download(source, filename):
        repo, revision, _ = source
        path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                               cache_dir=cache)
        downloads[repo + "/" + filename] = {"revision": revision, "sha256": sha256(path)}
        return Path(path)

    # Record source cards with the actual selected licenses and revisions.
    for source in (CV, KO):
        download(source, "README.md")

    def cap(split, lang):
        hours = args.hours if split == "train" else args.dev_hours
        return 3600 * hours * (args.english_ratio if lang == "en" else 1)

    def keep(split, lang, sample_id, speaker, text, audio, rate, source):
        if seconds[(split, lang)] >= cap(split, lang):
            return
        key = text_key(text)
        if not key or key in blocked_text or key in seen_text:
            rejected["text_duplicate_or_test"] += 1
            return
        if split == "train" and speaker in dev_speakers:
            rejected["dev_speaker"] += 1
            return
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != 16000:
            from math import gcd
            factor = gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // factor, rate // factor)
        duration = len(audio) / 16000
        if not (1.0 <= duration <= 15.0) or not np.isfinite(audio).all():
            rejected["audio_length_or_nonfinite"] += 1
            return
        path = wav_dir / f"{lang}-{sample_id}.wav"
        sf.write(path, audio, 16000, subtype="PCM_16")
        digest = sha256(path)
        if digest in seen_hash or digest in blocked_hash:
            path.unlink()
            rejected["audio_duplicate_or_test"] += 1
            return
        seen_hash.add(digest)
        seen_text.add(key)
        if split == "dev":
            dev_speakers.add(speaker)
        row = {"id": f"{lang}/{sample_id}", "language": lang, "speaker": speaker,
               "audio": str(path.relative_to(args.output)), "text": text,
               "seconds": duration, "sha256": digest,
               "source": source[0], "source_revision": source[1], "license": source[2]}
        rows[split].append(row)
        seconds[(split, lang)] += duration

    # Korean CV17's training split is only ~14 MB. Zeroth provides a useful
    # larger pool, with deterministic speaker-held-out development samples.
    import pyarrow.parquet as pq
    ko_files = sorted(p for p in HfApi().list_repo_files(KO[0], repo_type="dataset", revision=KO[1])
                      if p.startswith("data/train-") and p.endswith(".parquet"))
    for filename in ko_files:
        pool = pq.read_table(download(KO, filename)).to_pylist()
        random.Random(args.seed).shuffle(pool)
        for item in pool:
            speaker = f"zeroth/{item['speaker_id']}"
            import hashlib
            dev = int(hashlib.sha256(speaker.encode()).hexdigest()[:8], 16) % 10 == 0
            split = "dev" if dev else "train"
            if seconds[(split, "ko")] >= cap(split, "ko"):
                continue
            raw = item["audio"]
            audio, rate = sf.read(io.BytesIO(raw["bytes"]), dtype="float32")
            keep(split, "ko", str(item["id"]).replace("/", "-"), speaker,
                 item["text"], audio, rate, KO)
        print("Korean seconds", dict((s, seconds[(s, 'ko')]) for s in rows), flush=True)
        if all(seconds[(s, "ko")] >= cap(s, "ko") for s in rows):
            break

    # Use official CV development/train splits. Reserve every dev speaker and
    # sentence before selecting training, even if only a small dev cap is used.
    all_files = HfApi().list_repo_files(CV[0], repo_type="dataset", revision=CV[1])
    for lang, locale in (("ja", "ja"), ("zh", "zh-CN"), ("en", "en")):
        metadata = {}
        for split in ("dev", "train"):
            tsv = download(CV, f"transcript/{locale}/{split}.tsv")
            metadata[split] = read_tsv(tsv)
        excluded_speakers = {f"cv17/{r['client_id']}" for r in metadata["dev"]}
        excluded_text = {text_key(r["sentence"]) for r in metadata["dev"]}
        for split in ("dev", "train"):
            pool = metadata[split]
            if split == "train":
                pool = [r for r in pool if f"cv17/{r['client_id']}" not in excluded_speakers
                        and text_key(r["sentence"]) not in excluded_text]
            random.Random(args.seed).shuffle(pool)
            # Select a generous seeded candidate pool. Decoding order is stable
            # archive order; record that limitation rather than claim uniform sampling.
            wanted = {Path(r["path"]).name: r for r in pool[:max(200, int(cap(split, lang) / 2), len(pool) // 10)]}
            prefix = f"audio/{locale}/{split}/"
            shards = sorted(p for p in all_files if p.startswith(prefix) and p.endswith(".tar"))
            for filename in shards:
                with tarfile.open(download(CV, filename)) as archive:
                    for member in archive:
                        item = wanted.get(Path(member.name).name)
                        if item is None or not member.isfile():
                            continue
                        with archive.extractfile(member) as stream:
                            audio, rate = sf.read(io.BytesIO(stream.read()), dtype="float32")
                        keep(split, lang, Path(item["path"]).stem, f"cv17/{item['client_id']}",
                             item["sentence"], audio, rate, CV)
                        if seconds[(split, lang)] >= cap(split, lang):
                            break
                print(lang, split, "seconds", seconds[(split, lang)], flush=True)
                if seconds[(split, lang)] >= cap(split, lang):
                    break
            if seconds[(split, lang)] < cap(split, lang) * 0.9:
                raise RuntimeError(f"insufficient usable {lang}/{split} data; increase candidate pool")

    validate_splits(rows["train"], rows["dev"], protected)
    for split in rows:
        random.Random(args.seed).shuffle(rows[split])
        path = args.output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[split]))
    atomic_json(args.output / "data.json", {
        "selection": {"hours": args.hours, "dev_hours": args.dev_hours,
                      "english_ratio": args.english_ratio, "seed": args.seed},
        "protected_manifest_sha256": sha256(args.protected_test),
        "train_sha256": sha256(args.output / "train.jsonl"),
        "dev_sha256": sha256(args.output / "dev.jsonl"),
        "seconds": {f"{s}/{l}": v for (s, l), v in seconds.items()},
        "clips": {s: dict(Counter(r["language"] for r in rows[s])) for s in rows},
        "rejected": dict(rejected), "downloads": downloads,
        "sampling": "seeded candidate pool, duration caps, archive order within CV shards",
    })
    print(json.dumps({"clips": {s: len(r) for s, r in rows.items()},
                      "seconds": sum(seconds.values())}), flush=True)


if __name__ == "__main__":
    main()
