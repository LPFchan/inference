#!/usr/bin/env python3
"""Run resumable paid STT comparisons against the frozen FLEURS manifest."""
from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import httpx

API = "https://openrouter.ai/api/v1/audio/transcriptions"
MODELS = {
    "mai-transcribe-2": {"id": "microsoft/mai-transcribe-2", "hourly_usd": 0.10,
        "provider_options": {"azure": {"diarization": {"enabled": False},
            "enhancedMode": {"modelOptions": {"transcribeStyle": "verbatim"}}}}},
    "scribe-v2": {"id": "elevenlabs/scribe-v2", "hourly_usd": 0.22,
        "provider_options": {"elevenlabs": {"diarize": False, "tag_audio_events": False}}},
}


def payload(model, language, audio):
    # Only audio, model settings and a known language go to the provider.
    return {"model": MODELS[model]["id"], "language": language,
            "response_format": "json", "input_audio": {
                "data": base64.b64encode(audio).decode("ascii"), "format": "wav"},
            "provider": {"options": MODELS[model]["provider_options"]}}


def read_completed(path, model, manifest):
    completed = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            clip = row["path"]
            if clip in completed or clip not in manifest or row["model"] != model:
                raise ValueError(f"Invalid or duplicate saved prediction: {path.name} {clip}")
            if any(row.get(key) != value for key, value in manifest[clip].items()):
                raise ValueError(f"Saved prediction differs from manifest: {path.name} {clip}")
            if not isinstance(row.get("hypothesis"), str):
                raise ValueError(f"Invalid saved transcript: {path.name} {clip}")
            completed[clip] = row
    return completed


async def run(args, key, transport=None):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".openrouter-benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.data / "manifest.jsonl"
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
        manifest = {row["path"]: row for row in rows}
        if len(manifest) != len(rows):
            raise ValueError("Duplicate paths in manifest")
        if args.limit:
            rows = [row for lang in ("en", "ko", "ja", "zh")
                    for row in sorted([r for r in rows if r["language"] == lang],
                        key=lambda r: hashlib.sha256(f"stt-20261020:{r['path']}".encode()).hexdigest())[:args.limit]]
        done = {}
        outputs = {}
        for model, spec in MODELS.items():
            prediction = args.output / f"{model}.jsonl"
            done[model] = read_completed(prediction, model, manifest)
            metadata = {"model": model, "model_id": spec["id"], "api": API,
                        "manifest_sha256": manifest_hash, "known_language_when_supported": True,
                        "response_format": "json", "provider_options": spec["provider_options"]}
            runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            meta_path = args.output / f"{model}.metadata.json"
            if meta_path.exists():
                previous = json.loads(meta_path.read_text())
                if any(previous.get(k) != v for k, v in metadata.items()):
                    raise ValueError(f"Cannot resume changed run settings: {model}")
                if runner_hash not in previous["runner_sha256"]:
                    previous["runner_sha256"].append(runner_hash)
                    meta_path.write_text(json.dumps(previous, indent=2) + "\n")
            elif done[model]:
                raise ValueError(f"Cannot resume predictions without metadata: {model}")
            else:
                metadata["started_at"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
                metadata["runner_sha256"] = [runner_hash]
                meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
            outputs[model] = prediction.open("a", buffering=1)
        ledger_path = args.output / "openrouter-attempts.ndjson"
        attempts = [json.loads(line) for line in ledger_path.read_text().splitlines()] if ledger_path.exists() else []
        spend = sum(row["accounted_cost_usd"] for row in attempts)
        reserved = 0.0
        queue = asyncio.Queue()
        for row in rows:
            for model in MODELS:
                if row["path"] not in done[model]:
                    queue.put_nowait((model, row))
        stop = asyncio.Event()
        failures = []
        print(f"{queue.qsize()} clips remaining; accounted spend ${spend:.6f}", flush=True)
        with ledger_path.open("a", buffering=1) as ledger:
            async with httpx.AsyncClient(headers={"Authorization": "Bearer " + key}, timeout=90,
                    transport=transport, limits=httpx.Limits(max_connections=args.workers)) as client:
                async def worker():
                    nonlocal spend, reserved
                    while not stop.is_set():
                        try:
                            model, row = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            return
                        estimated = row["seconds"] / 3600 * MODELS[model]["hourly_usd"]
                        audio = (args.data / row["path"]).read_bytes()
                        if hashlib.sha256(audio).hexdigest() != row["sha256"]:
                            failures.append(f"WAV hash mismatch: {row['path']}")
                            stop.set()
                            return
                        body = payload(model, row["language"], audio)
                        for attempt in range(1, 5):
                            if spend + reserved + estimated > args.max_cost_usd:
                                failures.append("Cost limit reached; saved predictions can be resumed")
                                stop.set()
                                return
                            reserved += estimated
                            before = time.perf_counter()
                            result = None
                            status = None
                            generation = None
                            error = None
                            retryable = False
                            try:
                                response = await client.post(API, json=body)
                                status = response.status_code
                                generation = response.headers.get("x-generation-id")
                                retryable = status == 429 or status >= 500
                                if response.is_success:
                                    result = response.json()
                                else:
                                    error = f"HTTP {status}"
                            except httpx.TransportError as exc:
                                error = type(exc).__name__
                                retryable = True
                            except ValueError:
                                error = "Invalid JSON response"
                            elapsed = time.perf_counter() - before
                            usage = result.get("usage", {}) if isinstance(result, dict) else {}
                            cost = usage.get("cost")
                            reported = isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0
                            cost = float(cost) if reported else estimated
                            reserved -= estimated
                            spend += cost
                            entry = {"model": model, "path": row["path"], "attempt": attempt,
                                     "at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                                     "status": status, "error": error, "generation_id": generation,
                                     "usage": usage, "cost_reported": reported, "accounted_cost_usd": cost}
                            ledger.write(json.dumps(entry) + "\n")
                            if isinstance(result, dict) and isinstance(result.get("text"), str):
                                prediction = {**row, "model": model, "hypothesis": result["text"],
                                              "batch_seconds": elapsed, "batch_samples": 1,
                                              "generation_id": generation, "api_response": result}
                                outputs[model].write(json.dumps(prediction, ensure_ascii=False) + "\n")
                                done[model][row["path"]] = prediction
                                n = len(done[model])
                                if n % 25 == 0 or args.limit:
                                    print(f"{model} {n}/{len(manifest)}; accounted ${spend:.6f}", flush=True)
                                break
                            if retryable and attempt < 4:
                                print(f"Retry {model} {row['path']}: {error}", flush=True)
                                await asyncio.sleep(min(30, 2 ** attempt))
                                continue
                            failures.append(f"{model} {row['path']}: {error or 'Missing transcript'}")
                            stop.set()
                            break
                try:
                    results = await asyncio.gather(*(worker() for _ in range(args.workers)), return_exceptions=True)
                    failures.extend(type(result).__name__ for result in results if isinstance(result, BaseException))
                finally:
                    for output in outputs.values():
                        output.close()
        summary = {"accounted_cost_usd": spend, "attempts": len(ledger_path.read_text().splitlines()),
            "completed": {model: len(saved) for model, saved in done.items()}, "failures": failures}
        (args.output / "openrouter-run.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
        if failures:
            raise RuntimeError("; ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Clips per language, 0 for full splits")
    parser.add_argument("--max-cost-usd", type=float, default=4.0, help="Reported costs plus conservative estimates for unreported attempts")
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0 or not math.isfinite(args.max_cost_usd) or args.max_cost_usd <= 0:
        parser.error("Invalid worker count, clip limit or cost limit")
    key = args.key_file.read_text().strip() if args.key_file else os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        parser.error("Set OPENROUTER_API_KEY or --key-file")
    asyncio.run(run(args, key))


if __name__ == "__main__":
    main()
