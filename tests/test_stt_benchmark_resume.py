"""Protect frozen-set integrity and inference settings across benchmark resumes."""
import importlib.util
import json
import hashlib
import sys
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


@pytest.mark.parametrize("extra", [[], ["--batch-size", "8"], ["--nemotron-lookahead", "6"]])
def test_completed_run_checks_requested_settings_before_loading_model(tmp_path, monkeypatch, capsys, extra):
    row = {"path": "en/0.wav", "reference": "reference", "language": "en"}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    predictions = tmp_path / "nemotron.jsonl"
    predictions.write_text(json.dumps({**row, "model": "nemotron", "hypothesis": ""}) + "\n")
    metadata = {"model": "nemotron", "hostname": bench.platform.node(),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "limit_per_language": 0, "batch_size": 4, "known_language_when_supported": True,
                "model_id": bench.MODEL_IDS["nemotron"],
                "revision": bench.MODEL_SPECS["nemotron"]["revision"], "num_lookahead_tokens": 13}
    (tmp_path / "nemotron.metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(sys, "argv", ["benchmark", "--data", str(tmp_path), "--output", str(tmp_path),
                                    "--model", "nemotron", *extra])
    # This test environment has no torch: either outcome must precede GPU imports.
    if extra:
        with pytest.raises(ValueError, match="different inference settings"):
            bench.main()
    else:
        bench.main()
        assert "Already complete" in capsys.readouterr().out


def test_cli_rejects_lookahead_missing_from_pinned_processor(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--data", "/unused", "--output", "/unused",
                                    "--model", "nemotron", "--nemotron-lookahead", "1"])
    with pytest.raises(SystemExit) as exc:
        bench.main()
    assert exc.value.code == 2
