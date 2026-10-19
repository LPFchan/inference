# RSH-20261020-001: EG2 and multilingual speech recognition
Opened: 2026-10-20 05-53-32 KST
Recorded by agent: codex-stt

## Question and outcome

Compare the original [Embedding Gemma Transcribe](https://eg2.haku.sh/) native
engine with the [open reproduction](https://github.com/john-rocky/eg2-transcribe),
Qwen3-ASR-1.7B, Voxtral Mini 4B Realtime, and Whisper large-v3 on the same
English, Korean, Japanese and Mandarin test audio. Host both EG2 pipelines on
Dumpling regardless of their benchmark rank.

Qwen had the lowest primary error rate in every language in this run. The
original EG2 pipeline achieved 15.56% English WER versus 20.97% for the open
reader: a 5.41 percentage-point reduction (25.8% relative). Both EG2 readers
are English-only; the original mostly returns empty CJK transcripts, while the
open reader emits English-vocabulary text. Neither provides useful CJK STT.
Whisper was second on all four primary metrics, followed by Voxtral.

Both EG2 variants are deployed at `https://transcribe.lost.plus` under Common
Auth, with all serving processes on Dumpling. Model discovery advertises only
English, and explicit unsupported language requests return 400. The original
adapter executes the released app's native engine and original reader weights;
the open variant executes the independent reproduction.

## Measured error rates

Every model is evaluated on the complete frozen four-language test set, with
no dropped clips or reference transcripts supplied during inference. Values
are percentages; lower is better. Primary metrics are WER for English/Korean
and CER for Japanese/Mandarin. Scores over 100% are possible when insertions
outnumber reference units.

| Pipeline | English WER | Korean WER | Japanese CER | Mandarin CER |
| --- | ---: | ---: | ---: | ---: |
| Original EG2, app 1.6 | 15.56% | 100.00% | 99.83% | 99.73% |
| Open EG2, reader v2 | 20.97% | 104.14% | 144.56% | 204.75% |
| Qwen3-ASR-1.7B | 3.71% | 11.25% | 5.31% | 7.21% |
| Voxtral Mini 4B Realtime | 12.61% | 15.50% | 9.17% | 10.27% |
| Whisper large-v3 | 4.72% | 14.00% | 6.10% | 7.94% |

CER for every language removes word-spacing differences, especially relevant
to Korean. It uses the same normalization policy as the primary scores.

| Pipeline | English CER | Korean CER | Japanese CER | Mandarin CER |
| --- | ---: | ---: | ---: | ---: |
| Original EG2, app 1.6 | 11.25% | 99.91% | 99.83% | 99.73% |
| Open EG2, reader v2 | 12.96% | 156.55% | 144.56% | 204.75% |
| Qwen3-ASR-1.7B | 1.68% | 4.55% | 5.31% | 7.21% |
| Voxtral Mini 4B Realtime | 8.78% | 5.80% | 9.17% | 10.27% |
| Whisper large-v3 | 2.22% | 4.94% | 6.10% | 7.94% |

Voxtral returned 63 empty hypotheses (English 39, Korean 1, Japanese 22,
Mandarin 1). They remain in the corpus and count as deletion errors. A separate
16-clip preflight found identical single-file and batch-four outputs across
all four languages; that check does not establish equivalence for every clip.
The complete run uses the pinned implementation and default delay described
below, without per-model tuning to this test set.

These are measurements from this run, not model-card scores. This is read
speech in FLEURS, not meetings, singing, code switching, noise or diarization.
Unknown pretraining overlap and model-specific training corpora are not
controlled. No accuracy average combines WER and CER across languages, and no
speed ranking compares the M1 against the Thor GPU.

## Dataset and scoring

[Google FLEURS on Hugging Face](https://huggingface.co/datasets/google/fleurs)
(CC-BY-4.0), complete `test` splits at the parquet-conversion revision
`168de341b3db6859a9bac1c50a2ef5e3b47647e0`:

| Language | Configuration | Clips | Audio hours |
| --- | --- | ---: | ---: |
| English | en_us | 647 | 1.774 |
| Korean | ko_kr | 382 | 1.335 |
| Japanese | ja_jp | 650 | 2.364 |
| Mandarin, simplified Chinese | cmn_hans_cn | 945 | 3.074 |

Total: 2,624 clips, about 8.55 hours. Embedded audio is decoded to mono 16 kHz
PCM16 WAV once and the identical files go to every model. References are the
dataset's `transcription` field. Frozen manifest SHA-256:
`0c91ba8e164d33576c091385b4ddde130e4220756fe845f57d8736c9f86dd92e`.

Normalization is NFKC, Unicode case folding, punctuation/symbols replaced by
spaces, then whitespace collapse. WER tokenizes on spaces; CER removes spaces.
Corpus substitutions + deletions + insertions are divided by corpus reference
length, rather than averaging per-clip rates. No Japanese/Chinese word
segmentation, numeral expansion, script conversion or language-specific
normalizer is applied. Thus these numbers need not match published leaderboards.

The preparation script records source-parquet and per-WAV hashes in
`source.json` and `manifest.jsonl`. Scoring retains edit counts and denominators
in the external `summary.json`.

## Model pins and inference

| Pipeline | Pin | Execution |
| --- | --- | --- |
| Original EG2 | App 1.6, original v5 manifest | Dumpling M1 8 GB, macOS 27, Core ML CPU/ANE, released quantized assets, 20 s windows, native CTC/scoring/formatting |
| Open EG2 | [CoreAI bundle](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI) `76865ae46793e05617c2a7420454b0c7eb8541a4`, reader v2; upstream `e6b7d5df02fcfe940c194d00ad5d055f958b9c6c` | Same M1, FP16 Core AI `gpu-ane`, upstream 30 s windowing/overlap |
| [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | `7278e1e70fe206f11671096ffdd38061171dd6e5` | Mangchi Thor CUDA, BF16, official qwen-asr 0.0.6, Transformers 4.57.6, known language, max_new_tokens 512 |
| [Voxtral Mini 4B Realtime 2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | `2769294da9567371363522aac9bbcfdd19447add` | Same Thor, BF16, Transformers 5.16.1, native offline-file generation, default 6 delay tokens / 480 ms, automatic language |
| [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) | `06f233fe06e710322aca913c1bc4249a0d71fce1` | Same Thor, FP16, Transformers 4.57.6 ASR pipeline, known language, transcribe task, num_beams 1, 30 s chunks |

Reference models use batches of four; Voxtral groups similar durations to
reduce padding. Native workers use one clip at a time.
Whisper's pipeline emits its standard seq2seq chunking caveat; the longest
clip is 31.12 seconds. Qwen's model and processor both load the local pinned
snapshot because its package does not forward `revision` to its processor.
Voxtral uses full file input through the documented native Transformers API;
this measures transcription accuracy, not live-stream latency. The native
readers have English vocabularies; their CJK rows measure unsupported-language
behavior. No `language` field is sent to those readers during benchmarking.

The original and open rows compare complete released pipelines with different
readers, vocabularies, precision and preprocessing. They do not isolate the
effect of quantization or an encoder architecture change.

## Original app reverse engineering

The app's source is not published, but its downloadable 1.6 executable exports
an `EG2Transcriber` Swift library and its public manifest provides the original
model assets. The headless adapter reconstructs its Swift interface and calls
the actual native decoder, encoder/reader, CTC scoring, word timing and
`HeadOnlyPunct`/`TextFormatter`. It does not imitate the original reader with
open reproduction weights. Desktop UI, microphone capture and hotkeys are
outside this headless benchmark.

A local copy changes Mach-O loader metadata from executable to dylib, adds an
install name in verified empty header padding, clears the executable-only PIE
flag, resolves the installed Sparkle framework and receives an ad-hoc signature.
No machine code or source app is changed. The recovered class-bound punctuation
protocol is necessary for its Swift ABI. The build rejects any other executable
hash. No original binary, derived library, model asset or dataset is in Git.

- App disk image: `https://haku-models.tslfiles.org/eg2demo/EG2-Transcribe-1.6.dmg`
- DMG SHA-256: `4ee020455bf3cd6d3184b6f8271af060cd37be0bacc3669cca9739d2a7805214`
- Executable SHA-256: `7bcdc0085df93f5e5538e54e943d82c9701172361c56ec6a4241e325fa182348`
- v5 manifest: `https://haku-models.tslfiles.org/eg2demo/v5/manifest.json`
- Manifest SHA-256: `c7c63ca8e67ede94aa8af94aefff22be85309d6a715a20ca41af677479cdc644`

The original has 229,457 English vocabulary entries and reads encoder layer
`l11`; its 20 s input is `[1,2000,128]`, encoder frames are `[1,500,1024]`,
and reader outputs are 768-dimensional word vectors plus blank probabilities.
The reader uses UInt6 palette indices with FP16 values. The open bundle has an
independently trained reader and a 200k English vocabulary.

## Reproduction and retained evidence

Use `scripts/stt-prepare-fleurs.py`, `stt-download-reference-models.py`,
`stt-benchmark.py` and `stt-score.py`; model pins are in
`etc/stt-benchmark-models.json`. The final benchmark and scoring source is
commit `142c3c5669d0269f1e735e0802a7d37d29d51aab`; the Mangchi runner hash
matches that committed file. Preparation needs pyarrow and soundfile; scoring
needs jiwer. Reference downloads use huggingface_hub, filelock and aria2c, with
verified SHA-256 blobs and resumable ranged transfers outside the live cache.
Metadata files use direct HF downloads rather than concurrent nested snapshot
pools, which hit tqdm's shared-lock race in this environment.

Qwen/Whisper and Voxtral use separate repo-local Python 3.12 environments because
qwen-asr pins Transformers 4.x while Voxtral needs 5.x. Both reuse the installed
CUDA PyTorch 2.14.0+cu130 stack. Its native Triton CUDA operations require
matching Python development headers (`python3.12-dev`) and a C compiler.
The retained `environment/` under both model-data roots contains package
freezes and hardware/runtime metadata; do not install into system Python.

On Dumpling, audio/provenance are in `~/models/stt/fleurs-test` and all raw
predictions, model metadata and scores in `~/models/stt/results`. Reference-model
copies and environment freezes are in `~/models/stt-benchmark` on Mangchi.
Native assets and the installed original app are under `~/models/stt` on Dumpling.

All five prediction files contain exactly 2,624 unique paths, with the same
language, configuration, index, ID, reference, duration and WAV hash as the
frozen manifest. The Mangchi WAV bytes were independently hashed against that
manifest. `validation.json` retains the completeness checks and empty-output
counts; `summary.json` retains full-precision scores and edit counts.

Raw prediction SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `embeddinggemma-2-original.jsonl` | `d4899f0858a453326a459128f9c4a87f9bd0c11dc2dff34a6160bbf203ad2a4d` |
| `embeddinggemma-2-open.jsonl` | `e3155b7acc7ff32fee3ec54429cc6ce41180ea2b68da8d6ad8fbba7ee7f500b6` |
| `qwen.jsonl` | `a6a4df7391ec5b7cc1242a0cfff08392efa83f55ed2622e22d5abf71d6d50ecb` |
| `voxtral.jsonl` | `d13a3b10d89e63a2c319084f3fb5a1c339ee3eadd8a2d2cba06125e117c1000b` |
| `whisper.jsonl` | `5d5b3a273cedc3e707d9686d604ba0418ab318f0d175b1121d475ea776316009` |

Serving and all auth/tunnel processes stay on Dumpling. The canonical operating
and deployment guide is [`../../deploy/dumpling/README.md`](../../deploy/dumpling/README.md).
