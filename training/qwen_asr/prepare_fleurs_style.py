#!/usr/bin/env python3
"""Prepare FLEURS train written-form targets and a frozen KSS Korean probe.

CPU only; no model is loaded. Pinned parquet files are downloaded into a
caller-owned cache and decoded one row group at a time. Held-out text is used
only through normalized exclusion keys and is never printed. FLEURS publishes
no speaker ids, so speakers stay empty exactly like the frozen validation rows.
"""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time

from common import atomic_json, load_manifests, read_jsonl, sha256, text_key, validate_splits
from prefetch import SOURCES

FLEURS = {"ko": "ko_kr", "ja": "ja_jp", "zh": "cmn_hans_cn", "en": "en_us"}
KSS = ("Bingsu/KSS_Dataset", "48fdfd7ab1dbc1a62e4e8a8b9f4c360259d51d3c", "cc-by-nc-sa-4.0")
DIGIT = re.compile(r"\d")
MIN_FREE = 100 * (1 << 30)
CODE = [Path(__file__), Path(__file__).with_name("common.py"), Path(__file__).with_name("prefetch.py"),
        Path(__file__).resolve().parents[2] / "scripts/stt-score.py"]


def memory_pressure_normal():
    if sys.platform != "darwin":
        return True
    level = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                           capture_output=True, text=True, check=True).stdout
    return level.strip() == "1"


def guard(output, patience=600):
    """Pause this preparation on memory pressure; stop it if pressure persists."""
    if shutil.disk_usage(output).free < MIN_FREE:
        raise RuntimeError("less than 100 GiB free; preparation stopped")
    deadline = time.monotonic() + patience
    while not memory_pressure_normal():
        if time.monotonic() > deadline:
            raise RuntimeError("sustained memory pressure; preparation stopped")
        time.sleep(15)


def fetch(repo, revision, filenames, cache):
    """Download pinned dataset files and verify them against the hub's LFS hashes."""
    from huggingface_hub import HfApi, hf_hub_download
    expected = {i.path: i.lfs.sha256 for i in
                HfApi().get_paths_info(repo, filenames, revision=revision, repo_type="dataset")}
    files = {}
    for name in filenames:
        path = Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision, cache_dir=cache))
        digest = sha256(path)
        if digest != expected[name]:
            raise ValueError(f"download hash mismatch: {name}")
        files[name] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size,
                       "source": repo, "revision": revision}
    return files


def validation_ids():
    """FLEURS validation source-sentence ids for every language, read without text."""
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    repo, revision, _ = SOURCES["fleurs"]
    ids = {}
    for lang, config in FLEURS.items():
        with HfFileSystem().open(f"datasets/{repo}@{revision}/{config}/validation/0000.parquet") as stream:
            ids[lang] = pq.read_table(stream, columns=["id"])["id"].to_pylist()
    return ids


def rows(path, columns, output):
    """Yield (row index, row) while holding at most one parquet row group."""
    import pyarrow.parquet as pq
    source = pq.ParquetFile(path)
    index = 0
    for group in range(source.num_row_groups):
        guard(output)
        table = source.read_row_group(group, columns=columns, use_threads=False)
        for batch in table.to_batches(max_chunksize=16):
            for item in batch.to_pylist():
                yield index, item
                index += 1
        del table


def write_wav(path, data):
    """Decode encoded audio to mono 16 kHz PCM16; return (seconds, sha256) or None."""
    import io
    from math import gcd
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, rate = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16000:
        factor = gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // factor, rate // factor)
    seconds = len(audio) / 16000
    if not 1 <= seconds <= 15 or not np.isfinite(audio).all():
        return None
    sf.write(path, audio, 16000, subtype="PCM_16")
    return seconds, sha256(path)


def write_jsonl(path, items):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items))


def heldout(dev_dir, protected_test):
    dev, protected = read_jsonl(dev_dir / "dev.jsonl"), read_jsonl(protected_test)
    text = {text_key(r["text"]) for r in dev} | {text_key(r["reference"]) for r in protected}
    return dev, protected, text - {""}, {r["sha256"] for r in dev + protected}


def build_train(output, sources, blocked_ids, dev_dir, protected_test, exclude, downloads):
    """FLEURS train with raw_transcription targets, excluding held-out ids/text/audio."""
    dev, protected, held_text, held_audio = heldout(dev_dir, protected_test)
    probe = [r for path in exclude for r in read_jsonl(path)]
    probe_text = {text_key(r["text"]) for r in probe}
    probe_audio = {r["sha256"] for r in probe}
    repo, revision, license_name = SOURCES["fleurs"]
    seen, kept, overlap, rejected, train_ids = set(), [], Counter(), Counter(), {}
    (output / "wav").mkdir(parents=True, exist_ok=True)
    for lang, path in sources.items():
        train_ids[lang] = set()
        for index, item in rows(path, ["id", "audio", "raw_transcription", "transcription"], output):
            train_ids[lang].add(item["id"])
            text = " ".join((item["raw_transcription"] or "").split())
            keys = {text_key(text), text_key(item["transcription"] or "")} - {""}
            flags = {"heldout_source_sentence_id": item["id"] in blocked_ids,
                     "heldout_text": bool(keys & held_text), "probe_text": bool(keys & probe_text)}
            wav = output / "wav" / f"fleurs-train-{lang}-{index:05d}.wav"
            audio = write_wav(wav, item["audio"]["bytes"])
            if audio:
                flags |= {"heldout_audio": audio[1] in held_audio, "probe_audio": audio[1] in probe_audio,
                          "train_audio_duplicate": audio[1] in seen}
            for name, hit in flags.items():
                overlap[f"{lang}/{name}"] += hit
            if not text_key(text):
                reason = "empty_text"
            elif not audio:
                reason = "duration_or_nonfinite"
            else:
                reason = next((name for name, hit in flags.items() if hit), None)
            if reason:
                rejected[f"{lang}/{reason}"] += 1
                wav.unlink(missing_ok=True)
                continue
            seen.add(audio[1])
            kept.append({"id": f"fleurs-train/{lang}/{index}", "language": lang, "speaker": "",
                         "audio": str(wav.relative_to(output)), "text": text, "seconds": audio[0],
                         "sha256": audio[1], "source": repo, "source_revision": revision,
                         "license": license_name, "source_sentence_id": item["id"]})
        print(json.dumps({"language": lang, "kept": sum(r["language"] == lang for r in kept)}), flush=True)
    validate_splits(kept, dev, protected)
    if {text_key(r["text"]) for r in kept} & probe_text or {r["sha256"] for r in kept} & probe_audio:
        raise ValueError("probe overlap survived exclusion")
    write_jsonl(output / "train.jsonl", kept)
    for row in dev:
        target = output / row["audio"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dev_dir / row["audio"], target)
    shutil.copy2(dev_dir / "dev.jsonl", output / "dev.jsonl")
    for row in kept + dev:
        if sha256(output / row["audio"]) != row["sha256"]:
            raise ValueError(f"changed prepared audio: {row['id']}")
    languages = sorted(train_ids)
    seconds = Counter()
    for row in kept:
        seconds[row["language"]] += row["seconds"]
    atomic_json(output / "data.json", {
        "train_sha256": sha256(output / "train.jsonl"), "dev_sha256": sha256(output / "dev.jsonl"),
        "protected_manifest_sha256": sha256(protected_test), "dev_source_data_sha256": sha256(dev_dir / "data.json"),
        "selection": {"recipe": "fleurs-train-written-v1", "target_field": "raw_transcription",
                      "seconds": [1, 15], "order": "language ko, ja, zh, en; source row order"},
        "train_clips": len(kept), "dev_clips": len(dev), "train_hours": sum(seconds.values()) / 3600,
        "clips_by_language": dict(Counter(r["language"] for r in kept)),
        "hours_by_language": {k: v / 3600 for k, v in seconds.items()},
        "rejected": dict(rejected), "overlap_flags": dict(overlap),
        "source_sentence_ids": {
            "train_distinct": {k: len(v) for k, v in train_ids.items()},
            "train_shared_across_languages": {f"{x}&{y}": len(train_ids[x] & train_ids[y])
                                              for i, x in enumerate(languages) for y in languages[i + 1:]},
            "blocked_heldout_ids": len(blocked_ids)},
        "speakers": "unknown: FLEURS publishes no speaker ids; empty like frozen validation. "
                    "Train/validation speaker disjointness is not established.",
        "exclusion_manifests": {str(p): sha256(p) for p in exclude},
        "downloads": downloads, "preparation_code_sha256": {p.name: sha256(p) for p in CODE},
    })
    return load_manifests(output, protected_test)


def build_probe(output, files, dev_dir, protected_test, pools, seed, per_cohort, reserve, downloads):
    """Fixed balanced KSS original_script probe, frozen before any model output."""
    import pyarrow.parquet as pq
    _, _, held_text, held_audio = heldout(dev_dir, protected_test)
    pool_rows = [r for path in pools for r in read_jsonl(path)]
    pool_text, pool_audio = {text_key(r["text"]) for r in pool_rows}, {r["sha256"] for r in pool_rows}
    candidates = []
    for name, path in files:
        table = pq.read_table(path, columns=["original_script", "duration"])
        for index, (text, seconds) in enumerate(zip(table["original_script"].to_pylist(),
                                                    table["duration"].to_pylist())):
            candidates.append((name, index, " ".join((text or "").split()), seconds))
    repeats = Counter(text_key(c[2]) for c in candidates)
    stats, eligible = Counter(), {"digit": [], "other": []}
    for candidate in candidates:
        key = text_key(candidate[2])
        reason = ("empty_text" if not key else "repeated_text" if repeats[key] > 1
                  else "heldout_text" if key in held_text else "pool_text" if key in pool_text
                  else "declared_duration" if not 1 <= candidate[3] <= 15 else None)
        if reason:
            stats[reason] += 1
            continue
        eligible["digit" if DIGIT.search(candidate[2]) else "other"].append(candidate)
    rng = random.Random(seed)
    ranked = {cohort: rng.sample(items, len(items)) for cohort, items in eligible.items()}
    wanted = {(c[0], c[1]): (cohort, rank, c) for cohort, items in ranked.items()
              for rank, c in enumerate(items[:per_cohort + reserve])}
    (output / "wav").mkdir(parents=True, exist_ok=True)
    accepted, seen = {"digit": [], "other": []}, set()
    for name, path in files:
        for index, item in rows(path, ["audio"], output):
            if (name, index) not in wanted:
                continue
            cohort, rank, candidate = wanted[(name, index)]
            wav = output / "wav" / f"kss-{Path(name).stem}-{index:05d}.wav"
            audio = write_wav(wav, item["audio"]["bytes"])
            reason = ("duration_or_nonfinite" if not audio else "heldout_audio" if audio[1] in held_audio
                      else "pool_audio" if audio[1] in pool_audio else "repeated_audio" if audio[1] in seen
                      else None)
            if reason:
                stats[reason] += 1
                wav.unlink(missing_ok=True)
                continue
            seen.add(audio[1])
            accepted[cohort].append((rank, {
                "id": f"kss/{Path(name).stem}/{index}", "language": "ko", "speaker": "kss/single-speaker",
                "audio": str(wav.relative_to(output)), "text": candidate[2], "seconds": audio[0],
                "sha256": audio[1], "source": KSS[0], "source_revision": KSS[1], "license": KSS[2],
                "cohort": cohort}))
    probe = []
    for cohort, items in accepted.items():
        items.sort(key=lambda pair: pair[0])
        if len(items) < per_cohort:
            raise ValueError(f"insufficient {cohort} clips for the requested cohort; check exclusions or raise --reserve")
        for _, row in items[per_cohort:]:
            (output / row["audio"]).unlink()
        probe += [row for _, row in items[:per_cohort]]
    write_jsonl(output / "probe.jsonl", probe)
    for row in probe:
        if sha256(output / row["audio"]) != row["sha256"]:
            raise ValueError(f"changed probe audio: {row['id']}")
    population = {k: len(v) for k, v in eligible.items()}
    atomic_json(output / "probe.json", {
        "probe_sha256": sha256(output / "probe.jsonl"), "use": "evaluation only; exclude from every adaptation pool",
        "cohorts": {"digit": r"original_script contains a Unicode \d character", "other": "no such character"},
        "sampling": "balanced fixed-size cohorts; not natural-population accuracy",
        "seed": seed, "per_cohort": per_cohort, "reserve": reserve,
        "clips": dict(Counter(r["cohort"] for r in probe)), "eligible_population": population,
        "eligible_digit_share": population["digit"] / max(1, sum(population.values())),
        "hours": sum(r["seconds"] for r in probe) / 3600, "source_rows": len(candidates),
        "rejected": dict(stats), "protected_manifest_sha256": sha256(protected_test),
        "dev_sha256": sha256(dev_dir / "dev.jsonl"),
        "train_pool_manifests": {str(p): sha256(p) for p in pools},
        "downloads": downloads, "preparation_code_sha256": {p.name: sha256(p) for p in CODE},
    })
    return probe


def verify_probe(output):
    meta = json.loads((output / "probe.json").read_text())
    if sha256(output / "probe.jsonl") != meta["probe_sha256"]:
        raise ValueError("changed probe manifest")
    probe = read_jsonl(output / "probe.jsonl")
    for row in probe:
        if sha256(output / row["audio"]) != row["sha256"]:
            raise ValueError(f"changed probe audio: {row['id']}")
    return probe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("probe", "train"))
    p.add_argument("output", type=Path)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True, help="data directory holding the frozen dev.jsonl")
    p.add_argument("--protected-test", type=Path, required=True)
    p.add_argument("--train-pool", type=Path, action="append", default=[], help="probe: train manifests to exclude")
    p.add_argument("--exclude", type=Path, action="append", default=[], help="train: evaluation-only manifests")
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--per-cohort", type=int, default=256)
    p.add_argument("--reserve", type=int, default=64)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard(a.output)
        if a.command == "train":
            if (a.output / "data.json").exists():
                load_manifests(a.output, a.protected_test)
                return
            repo, revision, _ = SOURCES["fleurs"]
            names = {lang: f"{config}/train/0000.parquet" for lang, config in FLEURS.items()}
            downloads = fetch(repo, revision, list(names.values()), a.cache)
            ids = validation_ids()
            blocked = {i for v in ids.values() for i in v} | {int(r["id"]) for r in read_jsonl(a.protected_test)}
            downloads["validation_id_counts"] = {k: len(v) for k, v in ids.items()}
            build_train(a.output, {lang: downloads[name]["path"] for lang, name in names.items()}, blocked,
                        a.dev, a.protected_test, a.exclude, downloads)
        else:
            if (a.output / "probe.json").exists():
                verify_probe(a.output)
                return
            from huggingface_hub import HfApi
            names = sorted(f for f in HfApi().list_repo_files(KSS[0], repo_type="dataset", revision=KSS[1])
                           if f.startswith("data/train-") and f.endswith(".parquet"))
            downloads = fetch(KSS[0], KSS[1], names, a.cache)
            build_probe(a.output, [(n, downloads[n]["path"]) for n in names], a.dev, a.protected_test,
                        a.train_pool, a.seed, a.per_cohort, a.reserve, downloads)
        print(json.dumps({"ready": str(a.output)}), flush=True)


if __name__ == "__main__":
    main()
