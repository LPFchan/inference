# EmbeddingGemma speech benchmark tooling

The inference repository contains native adapters and a reproducible speech
recognition comparison. Model assets, frozen audio, predictions and scores
are retained under `~/models/stt` on Dumpling. GPU-reference copies and
benchmark evidence are retained under `~/models/stt-benchmark` on Mangchi.

Measured results and exact provenance:

- [Original/open EG2, Qwen, Voxtral and Whisper](../../records/research/RSH-20261020-001-eg2-multilingual-stt.md)
- [Seven-pipeline comparison with MAI-Transcribe 2 and Scribe v2](../../records/research/RSH-20261020-002-openrouter-stt.md)

## Benchmark tools

- `scripts/stt-prepare-fleurs.py` freezes the four FLEURS test splits into a
  manifest and mono 16 kHz WAVs with per-file SHA-256 hashes.
- `scripts/stt-download-reference-models.py` prepares the pinned GPU model
  snapshots from `etc/stt-benchmark-models.json`.
- `scripts/stt-benchmark.py` runs local references or a native worker backend
  and resumes saved JSONL predictions by manifest path.
- `scripts/stt-benchmark-openrouter.py` runs MAI-Transcribe 2 and Scribe v2,
  preserving API request IDs, costs and completed predictions.
- `scripts/stt-score.py` computes corpus WER/CER with shared normalization.

Use a repo-local `.venv`. Preparation needs pyarrow and soundfile; scoring
needs jiwer. Hosted inference needs httpx and a privately supplied OpenRouter
key. The reference-model downloader needs huggingface_hub, filelock and
aria2c. Its resumable transfers are verified before installation in the cache.

English/Korean use WER; Japanese/Mandarin use CER as the primary metric.
All four languages also have CER. Normalization is NFKC, case folding,
punctuation/symbol replacement with spaces, whitespace collapse and
whitespace-free character scoring. References are excluded from model input.

GPU references use batches of four. Voxtral groups clips by duration; native
workers use one file at a time. Qwen and Whisper use qwen-asr 0.0.6 /
Transformers 4.57.6; Voxtral uses Transformers 5.16.1 and mistral-common 1.12.0
in a separate repo-local environment. Native Triton CUDA operations require a
C compiler and matching Python development headers. Environment freezes and
request metadata accompany the retained results.

## Native adapters

Requires macOS 27 and Xcode 27. `stt/build.sh` compiles both headless workers;
`stt/README.md` describes their JSON-lines interface.

The original adapter reconstructs the exposed Swift library interface of app
1.6. `stt/prepare-original.py` pins the executable hash, changes Mach-O loader
metadata to a locally loadable library, and signs that local copy. Machine
code and the source app stay intact. The headless worker calls the original
preprocessing, encoder/reader, candidate scoring, CTC readout, word timing,
and punctuation. The app, weights and derived library stay outside Git.

The original app disk image is
`https://haku-models.tslfiles.org/eg2demo/EG2-Transcribe-1.6.dmg`, SHA-256
`4ee020455bf3cd6d3184b6f8271af060cd37be0bacc3669cca9739d2a7805214`.
Retain its license, NOTICE and Gemma policy alongside the downloaded app.

`stt/open/` preserves the Apache-2.0 audio and pipeline sources from
`john-rocky/eg2-transcribe` at
`e6b7d5df02fcfe940c194d00ad5d055f958b9c6c`, with its LICENSE. The open reader
bundle is `mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI` revision
`76865ae46793e05617c2a7420454b0c7eb8541a4` (reader v2). Both EG2 readers use
English vocabularies; their CJK scores measure unsupported-language behavior.
