"""Protect frozen-set integrity and inference settings across benchmark resumes."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("stt_benchmark", Path(__file__).parents[1] / "scripts/stt-benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_resume_preserves_empty_predictions(tmp_path):
    row = {"path": "ko/0.wav", "language": "ko", "reference": "reference", "sha256": "hash", "seconds": 1.2}
    path = tmp_path / "nemotron.jsonl"
    path.write_text(json.dumps({**row, "model": "nemotron", "hypothesis": ""}) + "\n")
    assert bench.completed_paths(path, [row], "nemotron") == {row["path"]}


@pytest.mark.parametrize("field,value", [("reference", "changed"), ("sha256", "changed"),
                                         ("seconds", 2.4), ("language", "ja"),
                                         ("model", "qwen"), ("hypothesis", None)])
def test_resume_rejects_changed_evidence(tmp_path, field, value):
    row = {"path": "ko/0.wav", "language": "ko", "reference": "reference", "sha256": "hash", "seconds": 1.2}
    prediction = {**row, "model": "nemotron", "hypothesis": "text", field: value}
    path = tmp_path / "nemotron.jsonl"
    path.write_text(json.dumps(prediction) + "\n")
    with pytest.raises(ValueError):
        bench.completed_paths(path, [row], "nemotron")


def test_resume_rejects_duplicates(tmp_path):
    row = {"path": "ko/0.wav", "reference": "ref"}
    prediction = json.dumps({**row, "model": "nemotron", "hypothesis": "text"}) + "\n"
    path = tmp_path / "nemotron.jsonl"
    path.write_text(prediction * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        bench.completed_paths(path, [row], "nemotron")


def test_resume_requires_same_checkpoint_and_chunk_setting(tmp_path):
    path = tmp_path / "metadata.json"
    metadata = {"revision": "pin", "num_lookahead_tokens": 13, "limit_per_language": 4}
    bench.preserve_metadata(path, metadata)
    bench.preserve_metadata(path, {**metadata, "limit_per_language": 0})
    for change in ({"revision": "another-pin"}, {"num_lookahead_tokens": 6}):
        with pytest.raises(ValueError, match="different inference settings"):
            bench.preserve_metadata(path, {**metadata, **change})
