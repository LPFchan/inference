#!/usr/bin/env python3
"""Download pinned EG2 runtimes without redistributing the original app assets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import subprocess
from pathlib import Path
import urllib.request

ORIGINAL = "https://haku-models.tslfiles.org/eg2demo/"
OPEN_REPO = "mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI"
OPEN_REVISION = "76865ae46793e05617c2a7420454b0c7eb8541a4"


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8.7.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def fetch(root, path, url, sha=None):
    target = root / path
    if target.is_file() and (sha is None or hashlib.sha256(target.read_bytes()).hexdigest() == sha):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    subprocess.run(["curl", "-fLsS", "--retry", "4", "--retry-all-errors", "--connect-timeout", "30",
                    "--max-time", "1800", "--continue-at", "-", url, "-o", str(temporary)], check=True)
    digest = hashlib.sha256()
    with temporary.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if sha and digest.hexdigest() != sha:
        raise ValueError(f"Checksum mismatch: {path}")
    temporary.replace(target)
    print(f"Downloaded {path} ({target.stat().st_size} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    original = args.root / "eg2-original-v5"
    manifest_bytes = get(ORIGINAL + "v5/manifest.json")
    if hashlib.sha256(manifest_bytes).hexdigest() != "c7c63ca8e67ede94aa8af94aefff22be85309d6a715a20ca41af677479cdc644":
        raise ValueError("The original v5 manifest changed; review it before downloading")
    manifest = json.loads(manifest_bytes)
    original.mkdir(parents=True, exist_ok=True)
    (original / "manifest.json").write_bytes(manifest_bytes)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda f: fetch(original, f["path"], ORIGINAL + f["url"], f["sha256"]), manifest["files"]))
    opened = args.root / "eg2-open-v2"
    base = f"https://huggingface.co/{OPEN_REPO}/resolve/{OPEN_REVISION}/"
    checksums = get(base + "SHA256SUMS").decode()
    files = []
    for line in checksums.splitlines():
        if not re.match(r"^[a-f0-9]{64}\s+", line):
            continue
        sha, path = line.split(maxsplit=1)
        path = path.lstrip("*")
        if path.startswith("macos/") or "/" not in path:
            files.append((path, sha))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda f: fetch(opened, f[0], base + f[0], f[1]), files))
    (opened / "SHA256SUMS").write_text(checksums)
    (opened / "source.json").write_text(json.dumps({"repo": OPEN_REPO, "revision": OPEN_REVISION}, indent=2))


if __name__ == "__main__":
    main()
