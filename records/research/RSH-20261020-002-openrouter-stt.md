# RSH-20261020-002: Hosted transcription on the shared FLEURS benchmark
Opened: 2026-10-20 12-27-17 KST
Recorded by agent: codex-stt

## Question and findings

Extend the original five-pipeline comparison with MAI-Transcribe 2 and ElevenLabs
Scribe v2 through OpenRouter on the same frozen test clips. Local-model settings
and native-engine provenance are in
[RSH-20261020-001](RSH-20261020-001-eg2-multilingual-stt.md).

MAI had the lowest error in all four languages for both primary metrics and
CER. Scribe improved primary Korean, Japanese and Mandarin scores over the
local reference models; Qwen retained a lower English WER than Scribe. Korean
ranking depends on the metric: Scribe's 8.63% WER is below Qwen's 11.25%, while
its 7.22% CER is above Qwen's 4.55%. Both views are retained.

All seven pipelines have exactly 2,624 predictions. The two API runs made
5,248 successful requests without retries or failed requests, for a reported
API usage cost of **$2.770335**. Eight initial multilingual smoke requests were
retained in the full results. Completed local-model predictions were reused.

## Measured error rates

Percentages, lower is better. Primary metrics are WER for English/Korean and
CER for Japanese/Mandarin. Empty outputs count as deletion errors; insertions
can produce error above 100%. The EG2 readers support English only.

| Pipeline | English WER | Korean WER | Japanese CER | Mandarin CER |
| --- | ---: | ---: | ---: | ---: |
| Original EG2, app 1.6 | 15.56% | 100.00% | 99.83% | 99.73% |
| Open EG2, reader v2 | 20.97% | 104.14% | 144.56% | 204.75% |
| Qwen3-ASR-1.7B | 3.71% | 11.25% | 5.31% | 7.21% |
| Voxtral Mini 4B Realtime | 12.61% | 15.50% | 9.17% | 10.27% |
| Whisper large-v3 | 4.72% | 14.00% | 6.10% | 7.94% |
| MAI-Transcribe 2 | 3.18% | 8.03% | 1.87% | 5.62% |
| ElevenLabs Scribe v2 | 4.62% | 8.63% | 3.20% | 7.07% |

CER across all four languages, with spaces removed:

| Pipeline | English CER | Korean CER | Japanese CER | Mandarin CER |
| --- | ---: | ---: | ---: | ---: |
| Original EG2, app 1.6 | 11.25% | 99.91% | 99.83% | 99.73% |
| Open EG2, reader v2 | 12.96% | 156.55% | 144.56% | 204.75% |
| Qwen3-ASR-1.7B | 1.68% | 4.55% | 5.31% | 7.21% |
| Voxtral Mini 4B Realtime | 8.78% | 5.80% | 9.17% | 10.27% |
| Whisper large-v3 | 2.22% | 4.94% | 6.10% | 7.94% |
| MAI-Transcribe 2 | 1.38% | 3.86% | 1.87% | 5.62% |
| ElevenLabs Scribe v2 | 2.94% | 7.22% | 3.20% | 7.07% |

Scribe returned nine empty transcripts (English 3, Korean 2, Japanese 2,
Mandarin 2); MAI returned none. The prior Voxtral row includes 63 empty
transcripts. No blank output was discarded or retried.

## Dataset and scoring

Complete [Google FLEURS](https://huggingface.co/datasets/google/fleurs) `test`
splits: `en_us` 647 clips, `ko_kr` 382, `ja_jp` 650, `cmn_hans_cn` 945.
The frozen parquet-conversion revision is
`168de341b3db6859a9bac1c50a2ef5e3b47647e0` (CC-BY-4.0). Identical mono 16 kHz
PCM16 WAVs total 30,769.12 seconds, or 8.546978 hours per pipeline.

Manifest SHA-256:
`0c91ba8e164d33576c091385b4ddde130e4220756fe845f57d8736c9f86dd92e`.
All seven sets were checked for unique paths and matching language, config,
index, ID, reference, duration and WAV hash. The API runner hashes each WAV
before sending it. The five earlier accuracy scores and edit counts are unchanged.

Scoring uses `scripts/stt-score.py` and jiwer 4.0.0. Normalization is NFKC,
Unicode case folding, punctuation/symbol replacement with spaces, then
whitespace collapse. WER splits on spaces; CER removes spaces. Corpus edit
totals are divided by corpus reference lengths. There is no numeral expansion,
Japanese word segmentation, script conversion or language-specific normalizer.
No average mixes WER and CER across languages.

These measurements cover read speech, not meetings, singing, noise, diarization
or mixed-language interaction. Training-data overlap is unknown. Scores are
measured here rather than taken from model cards or published leaderboards.

## Hosted inference settings

Requests completed on 2026-10-20 between 12:14:25 and 12:21:43 KST. Endpoint:
`https://openrouter.ai/api/v1/audio/transcriptions`, JSON with base64 WAV input,
known ISO language (`en`, `ko`, `ja`, `zh`) and `response_format=json`.
Each request contains one clip. The full run used 16 concurrent requests
shared across both models. No reference text, vocabulary hints or transcript
prompts were sent.

| Model alias | Provider in saved endpoint snapshot | Provider options sent |
| --- | --- | --- |
| `microsoft/mai-transcribe-2` | Azure | `diarization.enabled=false`, `enhancedMode.modelOptions.transcribeStyle=verbatim` |
| `elevenlabs/scribe-v2` | ElevenLabs | `diarize=false`, `tag_audio_events=false` |

These are unversioned hosted aliases: the underlying weight revision is not
exposed or pinned. Endpoint snapshots, exact request options, response usage
and generation IDs are retained for audit. Repeating an alias later can reach
a changed model. Request options follow the [OpenRouter STT guide](https://openrouter.ai/docs/guides/overview/multimodal/stt),
[Microsoft MAI documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe?pivots=programming-language-rest)
and [Scribe API reference](https://elevenlabs.io/docs/api-reference/speech-to-text/convert).

The measurement runner is `scripts/stt-benchmark-openrouter.py` at commit
`c273346eb151cc493b69e76e37f3068bc4f2359f`, frozen SHA-256
`77a678dc62f75cc43c15f800af1b73fb22f145c4d5d1ece04b6f03746e1b8d3a`.
Resume uses saved manifest paths, preserves empty text, checks request settings,
locks concurrent runs and logs request attempts. A $4 conservative usage limit
was applied. Credentials are excluded from the evidence.

The earlier pipelines ran on Dumpling M1 and Mangchi Thor as described in
RSH-20261020-001. No speed ranking compares those machines with hosted API
infrastructure. The Dumpling serving installation is unchanged.

## Observed API usage cost

| Model | Reported audio seconds | Endpoint rate observed | Sum of response `usage.cost`, USD |
| --- | ---: | --- | ---: |
| MAI-Transcribe 2 | 32,040 | $0.10/hour | $0.890000 |
| Scribe v2 | 30,769.12 | $0.22/hour | $1.880335 |
| Total | | | **$2.770335** |

MAI's reported seconds equal each clip's duration rounded upward to a whole
second, explaining its higher cost than the raw-duration estimate. Scribe
reports the original duration. These are returned API usage costs, excluding
any taxes or credit-purchase fees. They are not an account-balance delta that
could include unrelated activity. All 5,248 attempt records reported a cost.
[MAI pricing](https://openrouter.ai/microsoft/mai-transcribe-2) and
[Scribe pricing](https://openrouter.ai/elevenlabs/scribe-v2) are live pages;
the saved endpoint snapshots capture the rates observed for this run.

## Retained evidence and reproduction

Dumpling retains WAVs and the manifest in `~/models/stt/fleurs-test`.
`~/models/stt/results` contains all seven JSONL prediction files, metadata,
`openrouter-attempts.ndjson`, `openrouter-run.json`, `validation.json` and
full-precision `summary.json`. The package freeze is
`~/models/stt/environment/openrouter.freeze.txt`. Endpoint snapshots and the
frozen measurement runner are in `~/models/stt/openrouter-20261020`.
Audio, weights, API responses and credentials are outside Git.

| New prediction file | SHA-256 |
| --- | --- |
| `mai-transcribe-2.jsonl` | `e98a81363079e57a24e372e9d55bd8afc7cf7fa6e8ebfd7a080ee097ee46167e` |
| `scribe-v2.jsonl` | `d77771cb7ed18ac8f84afb16182750094e3038411fa15dbba14d0c77a9979ad3` |

Use a repo-local `.venv` with httpx and jiwer, set `OPENROUTER_API_KEY` privately
and run:

```sh
.venv/bin/python scripts/stt-benchmark-openrouter.py \
  --data "$HOME/models/stt/fleurs-test" \
  --output "$HOME/models/stt/results" --workers 16 --max-cost-usd 4
.venv/bin/python scripts/stt-score.py "$HOME/models/stt/results"
```

The runner skips saved clips. Calls on absent clips incur fresh charges;
use a separate output directory to deliberately measure a new run.
