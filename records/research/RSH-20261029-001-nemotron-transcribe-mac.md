# RSH-20261029-001: Nemotron Transcribe INT4 on the 24 GB Mac
Opened: 2026-10-29 03-24-07 KST
Recorded by agent: codex-nemotron-transcribe

## Question and outcome

Can the public [NVIDIA Nemotron 3.5 Transcribe 30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) run usefully on a 24 GB M4 Pro after quantization, and how does it compare with the eight existing measured pipelines?

The public release runs on the Mac as a 19.26 GB MLX INT4 export and now serves authenticated transcription. On this corpus, it improves Japanese CER over Qwen3-ASR-1.7B (4.13% versus 5.31%), while English WER, Korean WER and Mandarin CER are higher. It has the lowest Japanese CER among the local pipelines tested; MAI-Transcribe 2 retains the lowest error in all four languages. These results establish this local implementation’s behavior, not the unquantized model’s accuracy.

All nine pipelines contain 2,624 unique predictions with identical frozen paths, references, durations and WAV SHA-256 fields. The original eight JSONLs are byte-identical to their retained baselines, and their accuracy and edit counts are unchanged. No previous model predictions were rerun.

## Measured accuracy

Complete test splits; percentages, lower is better. Primary metrics use WER for English/Korean and CER for Japanese/Mandarin.

| Model | English WER | Korean WER | Japanese CER | Mandarin CER |
| --- | ---: | ---: | ---: | ---: |
| Original EG2, app 1.6 | 15.56% | 100.00% | 99.83% | 99.73% |
| Open EG2, reader v2 | 20.97% | 104.14% | 144.56% | 204.75% |
| Qwen3-ASR-1.7B | 3.71% | 11.25% | 5.31% | 7.21% |
| Nemotron Transcribe 30B-A4B, MLX INT4 | 5.35% | 13.77% | 4.13% | 8.52% |
| Nemotron ASR Streaming 0.6B | 10.46% | 25.32% | 13.34% | 19.20% |
| Voxtral Mini 4B Realtime | 12.61% | 15.50% | 9.17% | 10.27% |
| Whisper large-v3 | 4.72% | 14.00% | 6.10% | 7.94% |
| MAI-Transcribe 2 | 3.18% | 8.03% | 1.87% | 5.62% |
| Scribe v2 | 4.62% | 8.63% | 3.20% | 7.07% |

CER removes spaces for all languages:

| Model | English CER | Korean CER | Japanese CER | Mandarin CER |
| --- | ---: | ---: | ---: | ---: |
| Original EG2, app 1.6 | 11.25% | 99.91% | 99.83% | 99.73% |
| Open EG2, reader v2 | 12.96% | 156.55% | 144.56% | 204.75% |
| Qwen3-ASR-1.7B | 1.68% | 4.55% | 5.31% | 7.21% |
| Nemotron Transcribe 30B-A4B, MLX INT4 | 3.54% | 6.42% | 4.13% | 8.52% |
| Nemotron ASR Streaming 0.6B | 6.21% | 8.92% | 13.34% | 19.20% |
| Voxtral Mini 4B Realtime | 8.78% | 5.80% | 9.17% | 10.27% |
| Whisper large-v3 | 2.22% | 4.94% | 6.10% | 7.94% |
| MAI-Transcribe 2 | 1.38% | 3.86% | 1.87% | 5.62% |
| Scribe v2 | 2.94% | 7.22% | 3.20% | 7.07% |

No average combines languages or mixes WER and CER. Original/open EG2 use English readers; CJK scores describe unsupported-language behavior. Empty hypotheses remain included as errors. Earlier methods and exact revisions are in [RSH-20261020-001](RSH-20261020-001-eg2-multilingual-stt.md), [RSH-20261020-002](RSH-20261020-002-openrouter-stt.md), and [RSH-20261020-003](RSH-20261020-003-nemotron-stt.md). The new experiment makes no paid transcription API calls; the earlier hosted runs' reported total remains $2.7703351111.

## Dataset and scoring

The complete [Google FLEURS](https://huggingface.co/datasets/google/fleurs) test splits at parquet revision `168de341b3db6859a9bac1c50a2ef5e3b47647e0`: EN 647, KO 382, JA 650, simplified Mandarin 945. Audio totals 30,769.12 seconds, approximately 8.55 hours, as retained mono 16 kHz PCM16 WAVs. Manifest SHA-256: `0c91ba8e164d33576c091385b4ddde130e4220756fe845f57d8736c9f86dd92e`.

All 2,624 WAV hashes were independently checked on the Mac before inference and after completion. `scripts/stt-score.py` with jiwer 4.0.0 uses NFKC, case folding, punctuation/symbol replacement with spaces and whitespace collapse. WER splits on spaces; CER removes spaces. Corpus edit counts are divided by corpus reference lengths. There is no numeral expansion, language-specific spelling normalization, Japanese segmentation or script conversion. The references are unchanged from the previous experiment.

## Public checkpoint and local quantization

| Setting | Value |
| --- | --- |
| Source | `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` |
| Revision | `a188fff16c4f129a046788c2ad69e6198079a102` |
| BF16 safetensors bytes | 67,496,013,734 |
| BF16 SHA-256 | `03142dde5176c1d4df21ee829642745a1d670ea4d79e1a4684c64d53195acff2` |
| License | OpenMDW-1.1; source license and third-party notices retained |
| Runtime port | [MLX Audio](https://github.com/Blaizzy/mlx-audio/tree/c1e429151c56ce4057c62709f958df998e301d3f), pinned `c1e429151c56ce4057c62709f958df998e301d3f` |
| Export | MLX affine INT4, group size 64, 164 quantized language-model modules |
| Floating-point components | Speech/speaker encoders and router gates; Mamba dynamics/state use FP32 |
| Omitted optional heads | MTP and CTC timestamp heads |
| Exported parameter bytes | 19,258,434,176 (19.26 GB, 17.94 GiB) |
| Quantization peak MLX allocation | 3,408,823,968 bytes (3.41 GB) |
| Exporter | `scripts/stt-quantize-nemotron.py`, measurement commit `2030e83`, merged as `7ac5c50` |

The exporter verifies the full public source checksum and all 1,253 retained tensor shapes against the pinned port, evaluates one tensor at a time, and writes bounded shards. Runtime startup rechecks every exported file hash and loads with `strict=True`. `conversion.json` retains the complete source/export manifest and exact omitted tensor names.

The community early-access INT4 upload used a different source-weight checksum. Its weights were excluded. The public BF16 source was downloaded and quantized locally. An initially slow HF-Xet transfer was stopped in favor of a resumable aria2 transfer; this changed transport only. No runtime/JIT workaround was introduced.

## Inference and Mac serving

| Setting | Value |
| --- | --- |
| Host | yeowoolmac, The Rest partition, Apple M4 Pro, 24 GB unified memory |
| OS / Python | macOS 15.7.1; Python 3.12.13 |
| Runtime | MLX 0.32.4; pinned MLX Audio; full package freeze retained |
| API alias | `nemotron-3.5-transcribe-30b-a4b-4bit` |
| Input | Complete file, one request at a time; known ISO language hint |
| Decoding | Greedy, temperature 0, reasoning disabled; prefill step 256; no repetition penalty |
| Output cap | 4,096 generated tokens; cap exhaustion returns an error |
| Allocator cache | 128 MiB |
| Limits | 600-second audio, 100 MiB upload, serialized execution and four admitted requests |

The prompt follows NVIDIA's FLEURS verbatim instruction, with the known language substituted for automatic recognition:

```text
Audio transcription. Language: {English|Korean|Japanese|Chinese Mandarin}, in its standard writing system. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization; numbers in their standard written form. Output: transcript only.
```

The system prompt is empty. References, expected transcripts and vocabulary hints are excluded from model input. No prompt or quantization sweep was selected using these test scores. This is full-file transcription, without live microphone streaming, timestamps or diarization output. The previous pipelines used their documented native decoders/request options; shared scoring does not make all decoding methods identical. Numeral and formatting differences are included in errors.

Sixteen seeded preflight clips and each language's longest clip were saved with the final settings. The full run resumed those 20 paths without repeating them. The longest clip is 31.12 seconds. The serving API and benchmark use the same persistent MLX worker and artifact fingerprint. The runner holds an exclusive JSONL lock, validates saved rows/settings and WAV hashes, and rejects a mismatched server artifact.

The full continuation ran on 2026-10-29 from 02:38:31 to 03:18:53 GMT+9 (40 minutes 22 seconds), after the 20 saved preflight clips. All 2,624 predictions completed with zero failed clips, inference retries or empty outputs. The largest output was 113 tokens, below the cap.

Across all saved clips, worker elapsed time (audio loading and generation, excluding worker startup and HTTP overhead) had a median of 0.832 seconds and empirical 95th percentile of 1.340 seconds. Maximum measured MLX allocation was 20,175,557,844 bytes (20.18 GB). The first full-run request after idle took 67.21 seconds; the following requests resumed ordinary subsecond-to-few-second processing. The initial preflight request took 33.91 seconds end to end, including worker initialization. These observations do not guarantee consistently short cold or idle latency.

The existing `iogpu.wired_limit_mb=23040` was retained. oMLX was stopped and disabled with operator authorization; its models/settings remain. The Mac retains its GUI workload and some system swap, so this is an occupied 24 GB machine measurement. There is no cross-model speed ranking across the different machines and hosted APIs. No unquantized run was performed, so quantization accuracy loss is unmeasured.

`https://transcribe.lost.plus` is live through a dedicated Mac tunnel and the inference Common Auth gateway (`chat-v1` token / `chat` browser visibility). Backend and gateway bind loopback 8791/8792. Three LaunchAgents keep them running. Public audio smokes passed in all four languages with exact preflight transcript matches; missing/invalid credentials return 401 and an unsupported language returns 400. JSON/text output and authenticated model discovery pass. No Grimoire deployment or model eviction was involved. Installation and limits are in [`deploy/yeowoolmac`](../../deploy/yeowoolmac/README.md).

## Exact new-model edit counts

| Language | Word S / D / I | Reference words | Character S / D / I | Reference characters |
| --- | ---: | ---: | ---: | ---: |
| EN | 464 / 93 / 223 | 14573 | 590 / 400 / 1488 | 69950 |
| KO | 490 / 160 / 132 | 5679 | 462 / 548 / 125 | 17680 |
| JA | — | — | 875 / 229 / 245 | 32648 |
| ZH | — | — | 1169 / 1619 / 249 | 35655 |

## Evidence and reproduction

The Mac retains full BF16 source and the INT4 export under `~/models/stt/nemotron-transcribe-20261029/{source,mlx-4bit}`. Its `~/models/stt/results` holds all nine JSONLs, metadata, full-precision `summary.json` and `stt-results-nine.html`. Matching predictions, chart and non-weight evidence are mirrored to Dumpling `~/models/stt` and Mangchi `~/models/stt-benchmark`. The source/export weight files remain on the Mac; reproducible source IDs, model metadata and complete export hashes are mirrored.

Evidence includes the frozen runner/worker/exporter/server with SHA-256 values; source and quantized metadata/licenses; environment freezes; hardware/OS settings; all preflight, download, quantization and benchmark logs; request metadata and per-clip timing/memory/token counts; `validation-nine.json`; independent Dumpling WAV/row/hash validation; and production HTTPS smoke results. Mangchi's mirrored nine JSONLs were independently checked against the recorded hashes. No model weights or audio are committed to Git.

New prediction JSONL SHA-256: `9d6ff7f47ecb50ca34bb4e77cd88330fb76d5141e65207ffe7683d61deffd9bf`. Deployment/conversion manifest SHA-256: `7018b1c16891127e389845494e3e762703bbc6605e5c1136610405e053378a1b`.

Measurement code `2030e83` was reviewed and merged as `7ac5c50` without inference changes. The frozen runner SHA-256 is `97046ff6a70f17df82f951b7d154ad0866864bb74d49363ffa497040b3b72730`; worker `f17bfc9b1d12ba31df8c048dc13c73dc36a833b54870cf33f3e0c42e14285dcc`; exporter `cd2e5baa03e05cfe6edc1c1dacbddb0b0d410f37f9036cb163b0b997be798983`. Twenty-six focused STT, auth and benchmark tests pass. No benchmark CI is configured.

```sh
PYTHONPATH=src .venv-stt/bin/python scripts/stt-benchmark.py \
  --data "$HOME/models/stt/fleurs-test" --output "$HOME/models/stt/results" \
  --model nemotron-3.5-transcribe-30b-a4b-4bit \
  --mlx-model-dir "$HOME/models/stt/nemotron-transcribe-20261029/mlx-4bit"
.venv-stt/bin/python scripts/stt-score.py "$HOME/models/stt/results"
```

The benchmark does not establish meeting, noise, singing, code-switching, diarization, translation or microphone-streaming quality. Training-data overlap is uncontrolled. These are measured results for this frozen corpus and local INT4 implementation; NVIDIA's published benchmark scores are not substituted for them.
