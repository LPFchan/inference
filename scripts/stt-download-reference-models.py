#!/usr/bin/env python3
"""Fetch pinned benchmark weights with resumable, parallel aria2 transfers."""
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from filelock import FileLock

MODELS = json.loads((Path(__file__).resolve().parents[1] / "etc/stt-benchmark-models.json").read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(item):
    name, spec = item
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{spec['id']}/tree/{spec['revision']}") as response:
        files = [file for file in json.load(response) if file["type"] == "file"
                 and any(fnmatch.fnmatch(file["path"], pattern) for pattern in spec["allow_patterns"])]
    weights = [file for file in files if file["path"].endswith(".safetensors")
               and any(fnmatch.fnmatch(file["path"], pattern) for pattern in spec["allow_patterns"])]
    cache = Path(HF_HUB_CACHE) / ("models--" + spec["id"].replace("/", "--"))
    metadata = [file["path"] for file in files if not file["path"].endswith(".safetensors")]
    snapshot_download(spec["id"], revision=spec["revision"], allow_patterns=metadata, max_workers=4)
    for file in weights:
        oid = file["lfs"]["oid"]
        blob = cache / "blobs" / oid
        lock = Path(HF_HUB_CACHE) / ".locks" / cache.name / (oid + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock)):
            if not blob.exists():
                # Ranged transfers become sparse files. Keep them outside blobs
                # so model loaders cannot mistake a partial file for a full one.
                transfer = cache / "transfers" / oid
                transfer.mkdir(parents=True, exist_ok=True)
                staged = transfer / "weight"
                subprocess.run([
                    "aria2c", "--continue=true", "--max-connection-per-server=16", "--split=16",
                    "--min-split-size=16M", "--file-allocation=none", "--auto-file-renaming=false",
                    "--max-tries=8", "--retry-wait=5", "--timeout=120", "--summary-interval=30",
                    "--console-log-level=warn", "--download-result=hide", "--dir=" + str(transfer),
                    "--out=weight", f"https://huggingface.co/{spec['id']}/resolve/{spec['revision']}/{file['path']}",
                ], check=True)
                if staged.stat().st_size != file["size"] or sha256(staged) != oid:
                    raise ValueError(f"Checksum mismatch: {name}/{file['path']}")
                blob.parent.mkdir(parents=True, exist_ok=True)
                staged.replace(blob)
            elif sha256(blob) != oid:
                raise ValueError(f"Corrupt cached weight: {name}/{file['path']}")
        hf_hub_download(spec["id"], file["path"], revision=spec["revision"])
    print(f"READY {name}", flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, MODELS.items()))
