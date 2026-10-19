import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from grimoire.stt_server import MODEL_NAMES, create_app


class Worker:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.process = None
        self.metadata = {"ready": True}
        self.calls = []

    async def start(self):
        pass

    async def stop(self):
        pass

    async def transcribe(self, path):
        assert Path(path).read_bytes() == b"test audio"
        self.calls.append(str(path))
        return {"text": "test words", "seconds": 0.1, "duration": 1.0, "words": []}


def test_upload_formats_and_cleanup():
    worker = Worker()
    with TestClient(create_app({MODEL_NAMES[0]: worker})) as client:
        for response_format in ("json", "verbose_json", "text"):
            result = client.post("/v1/audio/transcriptions", data={"model": MODEL_NAMES[0], "response_format": response_format},
                                 files={"file": ("../../recording.wav", b"test audio")})
            assert result.status_code == 200, result.text
            assert result.headers["x-stt-model"] == MODEL_NAMES[0]
            assert result.text == "test words" if response_format == "text" else result.json()["text"] == "test words"
    assert len(worker.calls) == 3
    assert all(not Path(path).exists() for path in worker.calls)


def test_invalid_request_never_runs_worker():
    worker = Worker()
    with TestClient(create_app({MODEL_NAMES[0]: worker})) as client:
        for data, expected in (({"model": "unknown"}, 404), ({"response_format": "srt"}, 400),
                               ({"stream": "true"}, 400), ({"language": "ko"}, 400)):
            result = client.post("/v1/audio/transcriptions", data=data, files={"file": ("audio.wav", b"test audio")})
            assert result.status_code == expected, result.text
        assert client.post("/v1/audio/transcriptions", data={"model": MODEL_NAMES[0]}).status_code == 400
    assert worker.calls == []
