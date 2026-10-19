#!/usr/bin/env python3
"""Score all predictions with one transparent Unicode normalization policy."""
import argparse
import json
from pathlib import Path
import unicodedata

import jiwer


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(char)[0] in {"P", "S"} else char for char in text)
    return " ".join(text.split())


def summarize(rows):
    references = [normalize(row["reference"]) for row in rows]
    hypotheses = [normalize(row["hypothesis"]) for row in rows]
    chars = jiwer.process_characters([value.replace(" ", "") for value in references],
                                    [value.replace(" ", "") for value in hypotheses])
    words = jiwer.process_words(references, hypotheses)
    return {"clips": len(rows), "audio_seconds": sum(row["seconds"] for row in rows),
            "cer": chars.cer, "character_substitutions": chars.substitutions,
            "character_deletions": chars.deletions, "character_insertions": chars.insertions,
            "reference_characters": chars.hits + chars.substitutions + chars.deletions,
            "wer": words.wer if rows[0]["language"] in {"en", "ko"} else None,
            "word_substitutions": words.substitutions, "word_deletions": words.deletions,
            "word_insertions": words.insertions,
            "reference_words": words.hits + words.substitutions + words.deletions,
            "wall_seconds": sum(row["batch_seconds"] / row["batch_samples"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = {}
    for path in sorted(args.directory.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        report[path.stem] = {language: summarize([row for row in rows if row["language"] == language])
                             for language in ["en", "ko", "ja", "zh"] if any(row["language"] == language for row in rows)}
    (args.directory / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
