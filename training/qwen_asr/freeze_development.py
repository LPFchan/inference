#!/usr/bin/env python3
"""Freeze FLEURS validation and a small seed train snapshot, excluding test."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import soundfile as sf

from common import atomic_json, read_jsonl, sha256, text_key, validate_splits
from prefetch import SOURCES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--seed-data", type=Path, required=True)
    p.add_argument("--protected-test", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    output = a.root / "data-seed"
    if (output / "data.json").exists():
        from common import load_manifests
        load_manifests(output, a.protected_test)
        print("Frozen development and seed data already verified", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / "wav").mkdir(exist_ok=True)
    protected = read_jsonl(a.protected_test)
    blocked = {text_key(r["reference"]) for r in protected}
    blocked_hashes = {r["sha256"] for r in protected}
    dev, source_files = [], {}
    for lang, config in (("en","en_us"),("ko","ko_kr"),("ja","ja_jp"),("zh","cmn_hans_cn")):
        repo, revision, license_name = SOURCES["fleurs"]
        file = Path(hf_hub_download(repo, f"{config}/validation/0000.parquet", repo_type="dataset", revision=revision, cache_dir=a.cache))
        source_files[config] = sha256(file)
        for i, row in enumerate(pq.read_table(file).to_pylist()):
            text = row["transcription"]
            if text_key(text) in blocked:
                continue
            audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            assert rate == 16000 and audio.ndim == 1
            wav = output / "wav" / f"fleurs-dev-{lang}-{i:05d}.wav"
            sf.write(wav, audio, rate, subtype="PCM_16")
            digest = sha256(wav)
            if digest in blocked_hashes:
                continue
            dev.append({"id":f"fleurs-validation/{lang}/{i}", "language":lang, "speaker":"",
                        "audio":str(wav.relative_to(output)), "text":text, "seconds":len(audio)/rate,
                        "sha256":digest, "source":repo, "source_revision":revision,
                        "license":license_name})
        print(json.dumps({"development_language":lang,"clips":sum(r['language']==lang for r in dev)}), flush=True)
    blocked |= {text_key(r["text"]) for r in dev}
    blocked_hashes |= {r["sha256"] for r in dev}
    train = []
    for row in read_jsonl(a.seed_data / "train.jsonl"):
        if text_key(row["text"]) in blocked or row["sha256"] in blocked_hashes:
            continue
        row = dict(row)
        source = a.seed_data / row["audio"]
        target = output / row["audio"]
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(source, target)
        train.append(row)
    validate_splits(train, dev, protected)
    for name, rows in (("train",train),("dev",dev)):
        (output/f"{name}.jsonl").write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    atomic_json(output/"data.json",{
        "train_sha256":sha256(output/"train.jsonl"), "dev_sha256":sha256(output/"dev.jsonl"),
        "protected_manifest_sha256":sha256(a.protected_test), "selection":"frozen FLEURS validation, starter train seed",
        "source_files":source_files, "clips": {"train":dict(Counter(r['language'] for r in train)),"dev":dict(Counter(r['language'] for r in dev))}})
    print(json.dumps({"train":len(train),"development":len(dev),"dev_sha256":sha256(output/'dev.jsonl')}),flush=True)


if __name__ == "__main__":
    main()
