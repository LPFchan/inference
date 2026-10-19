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
MODEL_SPECS = json.loads((Path(__file__).resolve().parents[1] / "etc/stt-benchmark-models.json").read_text())
MODEL_IDS = {name: spec["id"] for name, spec in MODEL_SPECS.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=[*MODEL_IDS, "embeddinggemma-2-original", "embeddinggemma-2-open"], required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8791/v1/audio/transcriptions")
    parser.add_argument("--batch-size", type=int, default=4, help="GPU batch size")
    parser.add_argument("--limit", type=int, default=0, help="Seeded sample per language; 0 evaluates the full test splits")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    manifest = args.data / "manifest.jsonl"
    samples = [json.loads(line) for line in manifest.read_text().splitlines()]
    if args.limit:
        # Hash order avoids treating the first N as a representative random sample.
        samples = [row for lang in LANGUAGE_NAMES for row in sorted(
            [row for row in samples if row["language"] == lang],
            key=lambda row: hashlib.sha256(f"stt-20261020:{row['path']}".encode()).hexdigest())[:args.limit]]
    args.output.mkdir(parents=True, exist_ok=True)
    prediction_file = args.output / f"{args.model}.jsonl"
    done = set()
    if prediction_file.exists():
        done = {json.loads(line)["path"] for line in prediction_file.read_text().splitlines()}
    samples = [row for row in samples if row["path"] not in done]
    if not samples:
        print("Already complete", flush=True)
        return
    metadata = {"model": args.model, "hostname": platform.node(), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "limit_per_language": args.limit, "batch_size": args.batch_size if args.model in MODEL_IDS else 1,
                "known_language_when_supported": True}
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
            processor = AutoProcessor.from_pretrained(MODEL_IDS[args.model], revision=revision)
            if args.model == "whisper":
                from transformers import AutoModelForSpeechSeq2Seq, pipeline
                model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_IDS[args.model], revision=revision,
                                                                 torch_dtype=torch.float16).to("cuda:0").eval()
                recognizer = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                                      feature_extractor=processor.feature_extractor, device="cuda:0",
                                      torch_dtype=torch.float16, chunk_length_s=30, batch_size=args.batch_size)
            else:
                from transformers import VoxtralRealtimeForConditionalGeneration
                model = VoxtralRealtimeForConditionalGeneration.from_pretrained(MODEL_IDS[args.model], revision=revision,
                                                                                torch_dtype=torch.bfloat16).to("cuda:0").eval()
        metadata["transformers"] = importlib.metadata.version("transformers")
    (args.output / f"{args.model}.metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Loaded {args.model}; {len(samples)} clips remaining", flush=True)
    with prediction_file.open("a", buffering=1) as output:
        for language in LANGUAGE_NAMES:
            group = [row for row in samples if row["language"] == language]
            if args.model == "voxtral":
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
                elif args.model == "voxtral":
                    import soundfile as sf
                    audio_inputs = []
                    for path in paths:
                        audio, rate = sf.read(path, dtype="float32")
                        assert rate == processor.feature_extractor.sampling_rate
                        audio_inputs.append(audio)
                    inputs = processor(audio_inputs, return_tensors="pt").to(model.device, dtype=model.dtype)
                    with torch.inference_mode():
                        outputs = model.generate(**inputs)
                    texts = processor.batch_decode(outputs, skip_special_tokens=True)
                else:
                    import httpx
                    with open(paths[0], "rb") as audio:
                        response = httpx.post(args.url, data={"model": args.model, "response_format": "verbose_json"},
                                              files={"file": ("audio.wav", audio, "audio/wav")}, timeout=1800)
                    response.raise_for_status()
                    texts = [response.json()["text"]]
                elapsed = time.perf_counter() - before
                if len(texts) != len(batch):
                    raise RuntimeError("Backend returned the wrong batch size")
                for row, text in zip(batch, texts):
                    output.write(json.dumps({**row, "model": args.model, "hypothesis": text,
                                             "batch_seconds": elapsed, "batch_samples": len(batch)}, ensure_ascii=False) + "\n")
                print(f"{args.model} {language} {min(start + len(batch), len(group))}/{len(group)} ({elapsed:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
