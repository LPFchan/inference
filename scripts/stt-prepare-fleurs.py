#!/usr/bin/env python3
"""Freeze FLEURS test audio and references, using HF's parquet conversion."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import subprocess

import pyarrow.parquet as pq
import soundfile as sf

LANGUAGES = {"en": "en_us", "ko": "ko_kr", "ja": "ja_jp", "zh": "cmn_hans_cn"}
REVISION = "168de341b3db6859a9bac1c50a2ef5e3b47647e0"


def prepare(root, language, config):
    url = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/{config}/test/0000.parquet"
    target = root / f"{config}.parquet"
    if not target.is_file():
        print(f"Downloading {config}", flush=True)
        temporary = target.with_suffix(".part")
        subprocess.run(["curl", "-fLsS", "--retry", "4", "--retry-all-errors", "--connect-timeout", "30",
                        "--max-time", "1800", "--continue-at", "-", url, "-o", str(temporary)], check=True)
        temporary.replace(target)
    table = pq.read_table(target)
    rows = []
    folder = root / config
    folder.mkdir(exist_ok=True)
    for index, row in enumerate(table.to_pylist()):
        audio = row["audio"]["bytes"]
        wav = folder / f"{index:05d}.wav"
        if not wav.exists():
            samples, sample_rate = sf.read(io.BytesIO(audio), dtype="float32")
            assert sample_rate == 16000 and samples.ndim == 1
            sf.write(wav, samples, sample_rate, subtype="PCM_16")
        info = sf.info(wav)
        rows.append({"language": language, "config": config, "index": index,
                     "id": row.get("id"), "path": str(wav.relative_to(root)),
                     "reference": row["transcription"], "seconds": info.duration,
                     "sha256": hashlib.sha256(wav.read_bytes()).hexdigest()})
    print(f"Prepared {config}: {len(rows)} clips", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(lambda item: prepare(args.root, *item), LANGUAGES.items()))
    rows = [row for group in groups for row in group]
    manifest = args.root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    provenance = {"repo": "google/fleurs", "parquet_revision": REVISION,
                  "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in args.root.glob("*.parquet")},
                  "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (args.root / "source.json").write_text(json.dumps(provenance, indent=2))
    print(f"Wrote {len(rows)} samples to {manifest}", flush=True)


if __name__ == "__main__":
    main()
