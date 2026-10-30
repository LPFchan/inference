#!/usr/bin/env python3
"""Freeze the translator glosses of the FLEURS development references as optional spans.

FLEURS raw KO/JA/ZH text keeps a name's original Latin spelling in brackets, for example
罗克汉普顿（Rockhampton）. The cleaned `transcription` the references use drops the brackets but
keeps the spelling, and readers may or may not say it. A gloss is a bracketed span of a KO/JA/ZH
raw transcription with a Latin letter and no Hangul, kana or Han character; `selection.py
--glosses` scores each one as optional. English parentheses are spoken English and stay scored.
"""
import argparse
import json
from pathlib import Path
import re

from common import atomic_json, normalize, read_jsonl, sha256
from selection import gloss_spans

CONFIGS = {"ko": "ko_kr", "ja": "ja_jp", "zh": "cmn_hans_cn"}
BRACKETS = re.compile(r"[(（]([^()（）]*)[)）]")
NATIVE = re.compile("[ᄀ-ᇿ぀-ヿ㄰-㆏㐀-䶿一-鿿가-힯豈-﫿]")


def glosses(raw, reference):
    """Normalized glosses of one raw transcription, in order; each must occur in the reference."""
    found = [normalize(span) for span in BRACKETS.findall(raw)
             if re.search("[A-Za-z]", span) and not NATIVE.search(span)]
    gloss_spans(normalize(reference), found)
    return found


def build(dev_manifest, snapshot):
    import pyarrow.parquet as pq
    files = {lang: snapshot / config / "validation/0000.parquet" for lang, config in CONFIGS.items()}
    raw = {lang: pq.read_table(path, columns=["transcription", "raw_transcription"]).to_pylist()
           for lang, path in files.items()}
    rows = {}
    for row in read_jsonl(dev_manifest):
        found = []
        if row["language"] in CONFIGS:
            source = raw[row["language"]][int(row["id"].rsplit("/", 1)[1])]
            if source["transcription"] != row["text"]:
                raise ValueError(f"{row['id']} does not match its FLEURS row")
            found = glosses(source["raw_transcription"], row["text"])
        rows[row["id"]] = {"language": row["language"], "reference": row["text"], "glosses": found}
    return {"rule": __doc__.split("\n\n", 1)[1].strip(), "dev_manifest_sha256": sha256(dev_manifest),
            "sources": {CONFIGS[lang]: sha256(path) for lang, path in files.items()}, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dev_manifest", type=Path, help="frozen development dev.jsonl")
    p.add_argument("snapshot", type=Path, help="google/fleurs snapshot directory of the manifest's revision")
    p.add_argument("output", type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = build(a.dev_manifest, a.snapshot)
    atomic_json(a.output, result)
    counts = {lang: sum(len(r["glosses"]) for r in result["rows"].values() if r["language"] == lang) for lang in CONFIGS}
    print(json.dumps({"output": str(a.output), "glosses": counts}))


if __name__ == "__main__":
    main()
