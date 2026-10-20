# RSH-20261020-003: Nemotron 3.5 on the frozen multilingual STT benchmark
Opened: 2026-10-20 13-37-43 KST
Recorded by agent: codex-stt

## Question and outcome

Does [NVIDIA Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
improve recognition accuracy over Qwen3-ASR-1.7B on the existing English,
Korean, Japanese and Mandarin FLEURS test set?

Nemotron has higher error than Qwen in all four primary metrics and in CER
for every language. On this frozen read-speech test, it does not meet the
criterion of improving Korean, Japanese and Mandarin recognition accuracy.

The new model completed all 2,624 clips. All eight prediction sets have the
same paths, frozen references, durations and WAV SHA-256 values. The previous
seven prediction files and their accuracy/edit-count summaries are unchanged.
The Dumpling STT service remains retired; this experiment runs only on Mangchi.

## Measured accuracy

Complete test splits; percentages, lower is better. English/Korean use WER;
Japanese/Mandarin use CER. Every empty output is included as deletion errors.

| Model | English WER | Korean WER | Japanese CER | Mandarin CER |
| --- | ---: | ---: | ---: | ---: |
| Original EG2, app 1.6 | 15.56% | 100.00% | 99.83% | 99.73% |
| Open EG2, reader v2 | 20.97% | 104.14% | 144.56% | 204.75% |
| Qwen3-ASR-1.7B | 3.71% | 11.25% | 5.31% | 7.21% |
| Nemotron 3.5 ASR 0.6B | 10.46% | 25.32% | 13.34% | 19.20% |
| Voxtral Mini 4B Realtime | 12.61% | 15.50% | 9.17% | 10.27% |
| Whisper large-v3 | 4.72% | 14.00% | 6.10% | 7.94% |
| MAI-Transcribe 2 | 3.18% | 8.03% | 1.87% | 5.62% |
| Scribe v2 | 4.62% | 8.63% | 3.20% | 7.07% |

CER for all four languages removes word-spacing differences, especially
relevant to Korean:

| Model | English CER | Korean CER | Japanese CER | Mandarin CER |
| --- | ---: | ---: | ---: | ---: |
| Original EG2, app 1.6 | 11.25% | 99.91% | 99.83% | 99.73% |
| Open EG2, reader v2 | 12.96% | 156.55% | 144.56% | 204.75% |
| Qwen3-ASR-1.7B | 1.68% | 4.55% | 5.31% | 7.21% |
| Nemotron 3.5 ASR 0.6B | 6.21% | 8.92% | 13.34% | 19.20% |
| Voxtral Mini 4B Realtime | 8.78% | 5.80% | 9.17% | 10.27% |
| Whisper large-v3 | 2.22% | 4.94% | 6.10% | 7.94% |
| MAI-Transcribe 2 | 1.38% | 3.86% | 1.87% | 5.62% |
| Scribe v2 | 2.94% | 7.22% | 3.20% | 7.07% |

Nemotron returned no empty hypotheses. Voxtral's 63 and Scribe's nine empty
outputs remain included in their scores. Both EG2 readers support English
only; their CJK rows measure unsupported-language behavior.

No accuracy average combines different languages or WER with CER. These are
measurements from the frozen audio, not model-card numbers. Earlier local
methods are documented in [RSH-20261020-001](RSH-20261020-001-eg2-multilingual-stt.md);
[RSH-20261020-002](RSH-20261020-002-openrouter-stt.md) records the hosted models,
request options and their $2.7703351111 total reported cost. This extension
makes no paid transcription API calls.

## Shared dataset and scorer

[Google FLEURS](https://huggingface.co/datasets/google/fleurs), CC-BY-4.0,
complete `test` splits at parquet revision
`168de341b3db6859a9bac1c50a2ef5e3b47647e0`: English 647, Korean 382,
Japanese 650 and Mandarin 945; 30,769.12 seconds total. Audio is the retained
mono 16 kHz PCM16 WAV corpus. Manifest SHA-256:
`0c91ba8e164d33576c091385b4ddde130e4220756fe845f57d8736c9f86dd92e`.
All 2,624 WAV hashes were rechecked on Mangchi for this extension.

`scripts/stt-score.py` with jiwer 4.0.0 applies NFKC, case folding,
punctuation/symbol replacement with spaces and whitespace collapse. WER splits
on spaces; CER removes spaces. Corpus edit counts are divided by corpus
reference lengths. There is no numeral expansion, spelling normalization,
Japanese segmentation or script conversion. The identical scorer and frozen
references preserve comparability with the earlier runs.

## Nemotron checkpoint and inference

| Setting | Value |
| --- | --- |
| Model | `nvidia/nemotron-3.5-asr-streaming-0.6b` |
| Revision | `ea30d66debe3740a08b573244286791d423d6b3e` |
| Weight SHA-256 | `9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174` |
| License declared by checkpoint | OpenMDW-1.1 |
| Host | Mangchi, NVIDIA Jetson AGX Thor, SM110 |
| Runtime | Python 3.12.3, PyTorch 2.14.0+cu130, Transformers 5.16.1 |
| Precision | FP32 weights, no quantization; runtime flags retained in `system.json` |
| Input | Complete WAV files through documented `AutoModelForRNNT` offline API |
| Language prompts | `en-US`, `ko-KR`, `ja-JP`, `zh-CN`, one language per batch |
| Right context | 13 subsampled frames; 1.12-second chunks, fixed before the run |
| Decode | Native greedy RNN-T, default maximum 10 symbols per encoder frame |
| Batch | Four clips, sorted by duration within each language |
| Output | Native `sequences` decoded with special tokens removed and repeated RNN-T tokens retained |
| Generation runner | `scripts/stt-benchmark.py`, commit `ebcc655`, SHA-256 `2dd575ebeb71b975912a60a8a789ea2843dd33958331d2425af4ed482dc3e024` |

References, vocabulary hints and expected transcripts are excluded from model
input. The native generation limit is derived from encoder length; no fixed
text-token cap or 30-second audio truncation is applied. This measures
file-transcription accuracy, not live microphone streaming latency.

A 16-clip preflight selected four durations per language, including each
language's shortest and longest clips. All single-file transcripts matched
the corresponding batch-four transcripts exactly. The longest clip is 31.12
seconds. This check establishes equivalence for those clips only. No settings
were selected by scoring the test set, and no alternate precision, chunk-size
or larger-batch accuracy sweep was run.

The full measurement ran on 2026-10-20 from 13:29:58 to 13:35:15 KST and
completed without failed clips or inference retries. Scoring independently on
Dumpling reproduces every accuracy/edit-count result from Mangchi. Canonical
summary timing sums use Python 3.12; Python 3.9 differs only in floating-point
accumulation of duration/timing totals.

## Evidence and reproduction

Dumpling retains the eight prediction files, metadata, full-precision
`summary.json`, `validation-eight.json` and the interactive
`stt-results-eight.html` under `~/models/stt/results`. Mangchi retains matching
predictions under `~/models/stt-benchmark/results`, the frozen checkpoint in
its HF cache, and `~/models/stt-benchmark/nemotron-20261020` containing the
frozen runner/model specs, checkpoint metadata, source hashes, package freeze,
system settings, preflight outputs, validation script and run timestamps.
Audio and previous model assets remain intact. No weights or audio are in Git.

Nemotron prediction JSONL SHA-256: `da86432a815e53bc9b4da00e35e0c0e6d21f724d8e8f6548a1295ad8387dcde0`.

With the pinned checkpoint downloaded using the model manifest and a repo-local
environment containing the documented runtime:

```sh
.venv/bin/python scripts/stt-benchmark.py \
  --data "$HOME/models/stt-benchmark/fleurs-test" \
  --output "$HOME/models/stt-benchmark/results" \
  --model nemotron --batch-size 4 --nemotron-lookahead 13
.venv/bin/python scripts/stt-score.py "$HOME/models/stt-benchmark/results"
```

The runner locks each model output, validates saved rows against the frozen
manifest and rejects changed inference settings on resume. Completed paths
are skipped. The seven earlier model predictions were reused without inference.
Eighteen benchmark regression tests pass.

This read-speech corpus does not establish meeting, singing, noisy-audio,
code-switching, diarization or live-stream quality. Unknown training-data
overlap is uncontrolled. Hardware and API infrastructure differ across the
full eight-model comparison, so there is no cross-model speed ranking.
