#!/usr/bin/env python3
"""Resumable FLEURS test benchmark; references are never passed to a model."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

LANGUAGE_NAMES = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}
NEMOTRON_LOCALES = {"en": "en-US", "ko": "ko-KR", "ja": "ja-JP", "zh": "zh-CN"}
MODEL_SPECS = json.loads((Path(__file__).resolve().parents[1] / "etc/stt-benchmark-models.json").read_text())
MODEL_IDS = {name: spec["id"] for name, spec in MODEL_SPECS.items()}
MLX_MODEL = "nemotron-3.5-transcribe-30b-a4b-4bit"


def completed_paths(prediction_file, samples, model):
    """Reject stale or duplicated predictions before resuming a measured run."""
    expected = {row["path"]: row for row in samples}
    done = set()
    if prediction_file.exists():
        for line in prediction_file.read_text().splitlines():
            row = json.loads(line)
            path = row["path"]
            if path in done or path not in expected:
                raise ValueError(f"Duplicate or unknown prediction path: {path}")
            if row.get("model") != model or not isinstance(row.get("hypothesis"), str):
                raise ValueError(f"Invalid prediction: {path}")
            if any(row.get(key) != value for key, value in expected[path].items()):
                raise ValueError(f"Prediction disagrees with frozen manifest: {path}")
            done.add(path)
    return done


def check_metadata(path, metadata, require_existing=False):
    if require_existing and not path.exists():
        raise ValueError("Saved predictions have no inference metadata")
    if path.exists():
        previous = json.loads(path.read_text())
        # A limited preflight may be extended to the full set without changing
        # the inference settings that produced its existing predictions.
        keys = set(metadata) - {"limit_per_language"}
        if any(previous.get(key) != metadata[key] for key in keys):
            raise ValueError("Existing benchmark metadata uses different inference settings")


def preserve_metadata(path, metadata):
    check_metadata(path, metadata)
    path.write_text(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=[*MODEL_IDS, "embeddinggemma-2-original", "embeddinggemma-2-open", MLX_MODEL], required=True)
    parser.add_argument("--mlx-model-dir", type=Path, help="Pinned MLX export used by the local transcription server")
    parser.add_argument("--url", default="http://127.0.0.1:8791/v1/audio/transcriptions")
    parser.add_argument("--batch-size", type=int, default=4, help="GPU batch size")
    parser.add_argument("--limit", type=int, default=0, help="Seeded sample per language; 0 evaluates the full test splits")
    parser.add_argument("--nemotron-lookahead", type=int, choices=[0, 3, 6, 13], default=13,
                        help="Nemotron right-context frames; 13 gives 1120 ms chunks")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    manifest = args.data / "manifest.jsonl"
    samples = [json.loads(line) for line in manifest.read_text().splitlines()]
    args.output.mkdir(parents=True, exist_ok=True)
    prediction_file = args.output / f"{args.model}.jsonl"
    # Hold the lock through model loading and inference, including resume checks.
    from filelock import FileLock
    run_lock = FileLock(str(args.output / f".{args.model}.lock"))
    run_lock.acquire(timeout=0)
    done = completed_paths(prediction_file, samples, args.model)
    metadata = {"model": args.model, "hostname": platform.node(), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "limit_per_language": args.limit, "batch_size": args.batch_size if args.model in MODEL_IDS else 1,
                "known_language_when_supported": True}
    if args.model in MODEL_IDS:
        metadata.update(model_id=MODEL_IDS[args.model], revision=MODEL_SPECS[args.model]["revision"])
    if args.model == "nemotron":
        metadata["num_lookahead_tokens"] = args.nemotron_lookahead
    if args.model == MLX_MODEL:
        if args.mlx_model_dir is None:
            parser.error("--mlx-model-dir is required for the MLX benchmark")
        from grimoire.stt_mlx_worker import prompt_for, MAX_TOKENS
        conversion = args.mlx_model_dir / "conversion.json"
        metadata.update(artifact_sha256=hashlib.sha256(conversion.read_bytes()).hexdigest(),
                        conversion=json.loads(conversion.read_text()), max_tokens=MAX_TOKENS,
                        prompts={lang: prompt_for(lang) for lang in LANGUAGE_NAMES},
                        temperature=0, enable_thinking=False, prefill_step_size=256)
    metadata_path = args.output / f"{args.model}.metadata.json"
    check_metadata(metadata_path, metadata, require_existing=bool(done))
    if args.limit:
        # Hash order avoids treating the first N as a representative random sample.
        samples = [row for lang in LANGUAGE_NAMES for row in sorted(
            [row for row in samples if row["language"] == lang],
            key=lambda row: hashlib.sha256(f"stt-20261020:{row['path']}".encode()).hexdigest())[:args.limit]]
    samples = [row for row in samples if row["path"] not in done]
    if not samples:
        print("Already complete", flush=True)
        return
    if args.model == MLX_MODEL:
        import httpx
        discovery = httpx.get(args.url.rsplit("/v1/", 1)[0] + "/v1/models", timeout=30)
        discovery.raise_for_status()
        active = next((item for item in discovery.json()["data"] if item["id"] == MLX_MODEL), {})
        if active.get("artifact_sha256") != metadata["artifact_sha256"]:
            raise ValueError("Server artifact differs from the frozen benchmark deployment")
    if args.model in MODEL_IDS:
        import torch
        revision = MODEL_SPECS[args.model]["revision"]
        metadata.update(model_id=MODEL_IDS[args.model], revision=revision, torch=torch.__version__, device=torch.cuda.get_device_name())
        if args.model == "qwen":
            from huggingface_hub import snapshot_download
            from qwen_asr import Qwen3ASRModel
            # qwen-asr forwards revision to the model, but not its processor.
            # A local pinned snapshot keeps both on the same revision offline.
            checkpoint = snapshot_download(MODEL_IDS[args.model], revision=revision, local_files_only=True)
            model = Qwen3ASRModel.from_pretrained(checkpoint, dtype=torch.bfloat16,
                                                device_map="cuda:0", max_inference_batch_size=args.batch_size, max_new_tokens=512)
        else:
            from transformers import AutoProcessor
            checkpoint = MODEL_IDS[args.model]
            if args.model == "nemotron":
                from huggingface_hub import snapshot_download
                checkpoint = snapshot_download(checkpoint, revision=revision, local_files_only=True,
                                               allow_patterns=MODEL_SPECS[args.model]["allow_patterns"])
            processor = AutoProcessor.from_pretrained(checkpoint, revision=revision)
            if args.model == "whisper":
                from transformers import AutoModelForSpeechSeq2Seq, pipeline
                model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_IDS[args.model], revision=revision,
                                                                 torch_dtype=torch.float16).to("cuda:0").eval()
                recognizer = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                                      feature_extractor=processor.feature_extractor, device="cuda:0",
                                      torch_dtype=torch.float16, chunk_length_s=30, batch_size=args.batch_size)
            elif args.model == "voxtral":
                from transformers import VoxtralRealtimeForConditionalGeneration
                model = VoxtralRealtimeForConditionalGeneration.from_pretrained(MODEL_IDS[args.model], revision=revision,
                                                                                torch_dtype=torch.bfloat16).to("cuda:0").eval()
            else:
                from transformers import AutoModelForRNNT
                processor.set_num_lookahead_tokens(args.nemotron_lookahead)
                model = AutoModelForRNNT.from_pretrained(checkpoint, revision=revision,
                                                        dtype=torch.float32).to("cuda:0").eval()
                metadata.update(dtype="float32", num_lookahead_tokens=args.nemotron_lookahead,
                                chunk_ms=processor.streaming_latency_ms,
                                language_locales=NEMOTRON_LOCALES, inference="offline_file_rnnt_greedy",
                                max_symbols_per_step=model.config.max_symbols_per_step)
        metadata["transformers"] = importlib.metadata.version("transformers")
    preserve_metadata(metadata_path, metadata)
    print(f"Loaded {args.model}; {len(samples)} clips remaining", flush=True)
    with prediction_file.open("a", buffering=1) as output:
        for language in LANGUAGE_NAMES:
            group = [row for row in samples if row["language"] == language]
            if args.model in {"voxtral", "nemotron"}:
                group.sort(key=lambda row: row["seconds"])
            batch_size = args.batch_size if args.model in MODEL_IDS else 1
            for start in range(0, len(group), batch_size):
                batch = group[start:start + batch_size]
                paths = [str(args.data / row["path"]) for row in batch]
                before = time.perf_counter()
                if args.model == "qwen":
                    results = model.transcribe(audio=paths, language=[LANGUAGE_NAMES[language]] * len(paths))
                    texts = [result.text for result in results]
                elif args.model == "whisper":
                    import soundfile as sf
                    audio_inputs = []
                    for path in paths:
                        samples_array, sample_rate = sf.read(path, dtype="float32")
                        assert sample_rate == 16000 and samples_array.ndim == 1
                        audio_inputs.append({"raw": samples_array, "sampling_rate": sample_rate})
                    results = recognizer(audio_inputs, generate_kwargs={"language": language, "task": "transcribe", "num_beams": 1})
                    texts = [result["text"] for result in results]
                elif args.model in {"voxtral", "nemotron"}:
                    import soundfile as sf
                    audio_inputs = []
                    for row, path in zip(batch, paths):
                        audio, rate = sf.read(path, dtype="float32")
                        assert rate == processor.feature_extractor.sampling_rate
                        if args.model == "nemotron":
                            if audio.ndim != 1 or hashlib.sha256(Path(path).read_bytes()).hexdigest() != row["sha256"]:
                                raise ValueError(f"Audio disagrees with frozen manifest: {path}")
                        audio_inputs.append(audio)
                    kwargs = {"language": NEMOTRON_LOCALES[language], "sampling_rate": 16000} if args.model == "nemotron" else {}
                    inputs = processor(audio_inputs, return_tensors="pt", **kwargs).to(model.device, dtype=model.dtype)
                    with torch.inference_mode():
                        outputs = model.generate(**inputs)
                    sequences = outputs.sequences if args.model == "nemotron" else outputs
                    texts = processor.batch_decode(sequences, skip_special_tokens=True)
                else:
                    import httpx
                    data = {"model": args.model, "response_format": "verbose_json"}
                    if args.model == MLX_MODEL:
                        if hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest() != batch[0]["sha256"]:
                            raise ValueError(f"Audio disagrees with frozen manifest: {paths[0]}")
                        data["language"] = language
                    with open(paths[0], "rb") as audio:
                        response = httpx.post(args.url, data=data,
                                              files={"file": ("audio.wav", audio, "audio/wav")}, timeout=1800)
                    response.raise_for_status()
                    if args.model == MLX_MODEL and response.headers.get("x-stt-artifact") != metadata["artifact_sha256"]:
                        raise ValueError("Server artifact differs from the frozen benchmark deployment")
                    texts = [response.json()["text"]]
                elapsed = time.perf_counter() - before
                if len(texts) != len(batch):
                    raise RuntimeError("Backend returned the wrong batch size")
                for row, text in zip(batch, texts):
                    detail = {"inference": response.json()} if args.model == MLX_MODEL else {}
                    output.write(json.dumps({**row, "model": args.model, "hypothesis": text,
                                             "batch_seconds": elapsed, "batch_samples": len(batch), **detail}, ensure_ascii=False) + "\n")
                print(f"{args.model} {language} {min(start + len(batch), len(group))}/{len(group)} ({elapsed:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
