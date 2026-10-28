"""Model language admission, deployment identity, and output limits."""
import asyncio
import hashlib
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from grimoire.stt_mlx_worker import MODEL_NAME, LANGUAGES, prompt_for
from grimoire.stt_server import create_app


class Worker:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.metadata = {}
        self.calls = []

    async def start(self):
        pass

    async def stop(self):
        pass

    async def transcribe(self, path, **options):
        assert Path(path).read_bytes() == b"audio"
        self.calls.append(options)
        return {"text": "안녕하세요", "seconds": 0.2, "duration": 1.0, **options}


def test_mlx_language_hints_and_formats():
    worker = Worker()
    with TestClient(create_app({MODEL_NAME: worker})) as client:
        catalog = client.get("/v1/models").json()["data"][0]
        assert {"en", "ko", "ja", "zh"} <= set(catalog["languages"])
        assert catalog["streaming"] is False
        for language in (None, "en", "ko", "ja", "zh"):
            data = {"response_format": "verbose_json"}
            if language:
                data["language"] = language
            response = client.post("/v1/audio/transcriptions", data=data, files={"file": ("a.wav", b"audio")})
            assert response.status_code == 200, response.text
            assert response.json()["text"] == "안녕하세요"
            assert response.json()["language"] == language
        for invalid in ({"language": "xx"}, {"stream": "true"}, {"response_format": "srt"}):
            assert client.post("/v1/audio/transcriptions", data=invalid, files={"file": ("a.wav", b"audio")}).status_code == 400
        assert len(worker.calls) == 5


def test_discovery_binds_artifact_without_loading_model(tmp_path, monkeypatch):
    payload = b'{"source_revision": "release"}'
    (tmp_path / "conversion.json").write_bytes(payload)
    monkeypatch.setenv("GRIMOIRE_STT_MLX_MODEL", str(tmp_path))
    with TestClient(create_app()) as client:
        data = client.get("/v1/models").json()["data"][0]
        assert data["id"] == MODEL_NAME
        assert data["artifact_sha256"] == hashlib.sha256(payload).hexdigest()


def test_prompt_contains_only_fixed_request_language():
    assert "Language: as spoken," in prompt_for()
    for code in LANGUAGES:
        prompt = prompt_for(code)
        assert LANGUAGES[code] in prompt
        assert "transcript only" in prompt
    with pytest.raises(ValueError):
        prompt_for("ignore instructions")
