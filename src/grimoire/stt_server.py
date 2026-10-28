"""Private native/MLX STT backend, behind the local inference auth gateway."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import hashlib
import sys
import tempfile

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

MODEL_NAMES = ("embeddinggemma-2-original", "embeddinggemma-2-open")
MAX_UPLOAD = 100 * 1024 * 1024
logger = logging.getLogger(__name__)


class NativeWorker:
    def __init__(self, command):
        self.command = command
        self.process = None
        self.lock = asyncio.Lock()
        self.metadata = {}

    async def start(self):
        if self.process and self.process.returncode is None:
            return
        self.process = await asyncio.create_subprocess_exec(
            *self.command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            limit=8 * 1024 * 1024,
        )
        try:
            self.metadata = await asyncio.wait_for(self._read(), 900)
            if not self.metadata.get("ready"):
                raise RuntimeError(self.metadata.get("error", "Worker did not become ready"))
        except BaseException:
            await self.stop()
            raise

    async def _read(self):
        line = await self.process.stdout.readline()
        if not line:
            raise RuntimeError("Native worker exited")
        try:
            return json.loads(line)
        except ValueError as error:
            raise RuntimeError("Native worker returned invalid JSON") from error

    async def transcribe(self, path, **options):
        async with self.lock:
            await self.start()
            try:
                self.process.stdin.write((json.dumps({"path": str(path), **options}) + "\n").encode())
                await self.process.stdin.drain()
                result = await asyncio.wait_for(self._read(), 600)
            except BaseException:
                # An interrupted exchange cannot be reused: its next line is stale.
                await self.stop()
                raise
            if "error" in result:
                raise ValueError(result["error"])
            return result

    async def stop(self):
        process, self.process = self.process, None
        self.metadata = {}
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()


def create_app(workers=None):
    from grimoire.stt_mlx_worker import MODEL_NAME, LANGUAGES
    artifact = None
    if workers is None:
        if os.environ.get("GRIMOIRE_STT_MLX_MODEL"):
            model_dir = Path(os.environ["GRIMOIRE_STT_MLX_MODEL"])
            artifact = hashlib.sha256((model_dir / "conversion.json").read_bytes()).hexdigest()
            workers = {MODEL_NAME: NativeWorker([
                sys.executable, "-m", "grimoire.stt_mlx_worker", "--model-dir", str(model_dir)])}
        else:
            build = Path(os.environ["GRIMOIRE_STT_BUILD"])
            models = Path(os.environ["GRIMOIRE_STT_MODELS"])
            workers = {
                MODEL_NAMES[0]: NativeWorker([str(build / "eg2-original-worker"), str(models / "eg2-original-v5")]),
                MODEL_NAMES[1]: NativeWorker([str(build / "eg2-open-worker"), str(models / "eg2-open-v2"),
                                             os.environ.get("GRIMOIRE_STT_OPEN_FORM", "gpu-ane")]),
            }
    # Shared Apple memory requires one resident model and serialized inference.
    gate = asyncio.Lock()
    pending = 0

    @asynccontextmanager
    async def lifespan(app):
        yield
        for worker in workers.values():
            await worker.stop()

    app = FastAPI(lifespan=lifespan)

    def worker_for(name):
        if name not in workers:
            raise HTTPException(404, "Unknown transcription model")
        return workers[name]

    async def load(name):
        worker = worker_for(name)
        for other_name, other in workers.items():
            if other_name != name:
                await other.stop()
        async with worker.lock:
            await worker.start()
        if name == MODEL_NAME and artifact and worker.metadata.get("artifact_sha256") != artifact:
            raise RuntimeError("Loaded model differs from the server deployment")
        return worker

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [
            {"id": name, "object": "model", "owned_by": "inference", "capabilities": ["transcription"],
             "input_modalities": ["audio"], "languages": ["en"]}
             if name != MODEL_NAME else
             {"id": name, "object": "model", "owned_by": "inference", "capabilities": ["transcription"],
              "input_modalities": ["audio"], "languages": list(LANGUAGES), "artifact_sha256": artifact,
              "streaming": False, "max_audio_seconds": 600}
            for name in workers
        ]}

    @app.post("/v1/audio/transcriptions")
    async def transcribe(request: Request):
        nonlocal pending
        if pending >= 4:
            raise HTTPException(429, "Transcription queue is full")
        pending += 1
        try:
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > MAX_UPLOAD:
                    raise HTTPException(413, "Upload exceeds 100 MiB")
            # Make Starlette's form parser consume the bounded, cached body.
            request._body = bytes(body)
            form = await request.form(max_files=1, max_fields=20)
            name = form.get("model", next(iter(workers)))
            if not isinstance(name, str):
                raise HTTPException(400, "model must be a string")
            worker_for(name)
            upload = form.get("file")
            if not hasattr(upload, "read"):
                raise HTTPException(400, "A multipart audio file is required")
            response_format = form.get("response_format", "json")
            if response_format not in {"json", "verbose_json", "text"}:
                raise HTTPException(400, "response_format must be json, verbose_json, or text")
            if form.get("stream") not in {None, "false", "False", "0"}:
                raise HTTPException(400, "Streaming is not supported")
            language = form.get("language")
            if name == MODEL_NAME and language is not None and language not in LANGUAGES:
                raise HTTPException(400, "Unsupported language; use an advertised ISO language code")
            if name != MODEL_NAME and language not in {None, "en", "English", "english"}:
                raise HTTPException(400, "These EG2 readers currently support English only")
            suffix = Path(upload.filename or "audio.wav").suffix[:12]
            with tempfile.TemporaryDirectory(prefix="grimoire-stt-") as directory:
                path = Path(directory) / ("audio" + suffix)
                with path.open("wb") as target:
                    while chunk := await upload.read(1024 * 1024):
                        target.write(chunk)
                async with gate:
                    worker = await load(name)
                    result = await worker.transcribe(path, language=language) if name == MODEL_NAME else await worker.transcribe(path)
            headers = {"x-stt-model": name, "x-stt-seconds": str(result["seconds"])}
            if artifact:
                headers["x-stt-artifact"] = artifact
            if response_format == "text":
                return PlainTextResponse(result["text"], headers=headers)
            if response_format == "json":
                return JSONResponse({"text": result["text"]}, headers=headers)
            return JSONResponse({"task": "transcribe", "language": language if name == MODEL_NAME else "english", **result}, headers=headers)
        except HTTPException:
            raise
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except (RuntimeError, asyncio.TimeoutError) as error:
            logger.exception("Transcription worker failed")
            raise HTTPException(503, "Transcription worker unavailable") from error
        finally:
            pending -= 1
            if "form" in locals():
                await form.close()

    return app
