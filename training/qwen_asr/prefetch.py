#!/usr/bin/env python3
"""Prefetch pinned, public expansion sources without starting GPU work."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import random
import shutil

from huggingface_hub import HfApi, hf_hub_download
from common import atomic_json, sha256

SOURCES = {
    "fleurs": ("google/fleurs", "168de341b3db6859a9bac1c50a2ef5e3b47647e0", "cc-by-4.0"),
    "zeroth": ("kresnik/zeroth_korean", "1fe937899f828af822293d05e086200946088bdf", "cc-by-4.0"),
    "cv17": ("fsicoli/common_voice_17_0", "8262c16bf297c87a9cd88c51997c4758ed7a8ba2", "cc0-1.0"),
    "yodas": ("espnet/yodas", "0cad6a0ebec75e945fef92084052be69457fe9d1", "cc-by-3.0"),
    "aishell": ("AISHELL/AISHELL-1", "bbe295d530192a4cd41644b711c9aecd087df653", "apache-2.0"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--japanese-shards", type=int, default=12)
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    with (a.root / ".prefetch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = [("fleurs", f"{c}/validation/0000.parquet") for c in ("en_us","ko_kr","ja_jp","cmn_hans_cn")]
        jobs += [(name, "README.md") for name in SOURCES if name != "fleurs"]
        for name in ("zeroth", "cv17", "yodas", "aishell"):
            repo, revision, _ = SOURCES[name]
            files = HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
            if name == "zeroth":
                jobs += [(name, f) for f in sorted(files) if f.startswith("data/train-") and f.endswith(".parquet")]
            elif name == "cv17":
                jobs += [(name, f) for f in files if f.startswith(("audio/ja/train/", "transcript/ja/", "audio/zh-CN/train/", "transcript/zh-CN/")) and (f.endswith(".tar") or f.endswith(("train.tsv", "dev.tsv")))]
            elif name == "yodas":
                audio = sorted(f for f in files if f.startswith("data/ja000/audio/") and f.endswith(".tar.gz"))
                random.Random(20261029).shuffle(audio)
                for f in audio[:a.japanese_shards]:
                    index = Path(f).name.removesuffix(".tar.gz")
                    jobs += [(name, f"data/ja000/text/{index}.txt"), (name, f"data/ja000/duration/{index}.txt"), (name, f)]
            else:
                jobs.append((name, "data_aishell/transcript/aishell_transcript_v0.8.txt"))
                jobs += [(name, f) for f in sorted(files) if f.startswith("data_aishell/wav/S") and f.endswith(".tar.gz") and int(Path(f).name[1:5]) <= 341]
        jobs = list(dict.fromkeys(jobs))
        catalog = {"sources": SOURCES, "jobs": jobs, "completed": {}, "status": "running"}
        path = a.root / "prefetch.json"
        if path.exists():
            previous = json.loads(path.read_text())
            for key, value in previous.get("completed", {}).items():
                if Path(value["path"]).is_file() and sha256(value["path"]) == value["sha256"]:
                    catalog["completed"][key] = value
        atomic_json(path, catalog)

        def fetch(job):
            name, filename = job
            if shutil.disk_usage(a.root).free < 100 * (1 << 30):
                raise RuntimeError("less than 100 GiB free; intake stopped without deleting anything")
            repo, revision, license_name = SOURCES[name]
            file = hf_hub_download(repo, filename, repo_type="dataset", revision=revision, cache_dir=a.cache)
            return name + "/" + filename, {"path": file, "sha256": sha256(file), "source": repo,
                                            "revision": revision, "license": license_name}

        pending = [j for j in jobs if j[0] + "/" + j[1] not in catalog["completed"]]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fetch, j): j for j in pending}
            catalog["errors"] = {}
            for f in as_completed(futures):
                job = futures[f]
                key = job[0] + "/" + job[1]
                try:
                    key, value = f.result()
                    catalog["completed"][key] = value
                except Exception as exc:
                    catalog["errors"][key] = str(exc)
                atomic_json(path, catalog)
                print(json.dumps({"completed": len(catalog["completed"]), "total": len(jobs),
                                  "file": key, "errors": len(catalog["errors"])}), flush=True)
        catalog["status"] = "complete" if not catalog["errors"] else "partial"
        atomic_json(path, catalog)


if __name__ == "__main__":
    main()
