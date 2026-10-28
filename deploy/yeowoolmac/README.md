# Mac transcription

Nemotron 3.5 Transcribe 30B-A4B runs on yeowoolmac's **The Rest** partition
(M4 Pro, 24 GB). The native MLX backend binds loopback `8791`; the inference
Common Auth gateway binds loopback `8792`, using the existing `chat-v1` token
scope and `chat` browser visibility. A dedicated `yeowoolmac-stt` Cloudflare
tunnel fronts `https://transcribe.lost.plus`. No Grimoire process is required.

## Model preparation

Use Python 3.12 and the repo-local `.venv-stt`, with
`requirements-stt.txt`. Download NVIDIA's public checkpoint at revision
`a188fff16c4f129a046788c2ad69e6198079a102` outside Git. Retain its license
and third-party notices. The BF16 safetensors SHA-256 must be
`03142dde5176c1d4df21ee829642745a1d670ea4d79e1a4684c64d53195acff2`.

```sh
.venv-stt/bin/python scripts/stt-quantize-nemotron.py \
  "$HOME/models/stt/nemotron-transcribe-20261029/source" \
  "$HOME/models/stt/nemotron-transcribe-20261029/mlx-4bit"
```

The export uses affine INT4, group size 64 for language-model matrices.
Speech/speaker encoders and router gates retain their floating-point weights;
Mamba state parameters retain FP32. It omits optional MTP and CTC timestamp
heads. Conversion checks every retained tensor shape against the pinned MLX
implementation before writing bounded shards. `conversion.json` records
source/export hashes. Worker startup verifies those hashes and strictly loads
the complete exported model.

The early-access community quant has different source weights and is not this
deployment. NVIDIA's NVFP4 format requires a compatible runtime; this Mac uses
MLX affine INT4.

## Installation and API

Stop competing oMLX models before loading this model. Its LaunchAgent can be
stopped with `launchctl bootout gui/$(id -u)/plus.lost.omlx` and disabled with
`launchctl disable gui/$(id -u)/plus.lost.omlx`. Preserve weights and settings.

Provision the dedicated tunnel using the existing helper, supplying the
Cloudflare key privately on stdin:

```sh
python3 deploy/dumpling/provision-cloudflare.py --tunnel-name yeowoolmac-stt
STT_MLX_MODEL="$HOME/models/stt/nemotron-transcribe-20261029/mlx-4bit" \
  sh deploy/yeowoolmac/install-stt.sh
```

The helper writes the tunnel token to `~/.config/inference/stt-tunnel-token`
with mode 0600. It refuses conflicting DNS records and preserves unrelated
Cloudflare rules. LaunchAgents keep the backend, gateway, and tunnel running;
logs are in `~/Library/Logs/inference-stt`.

```sh
curl https://transcribe.lost.plus/v1/audio/transcriptions \
  -H "Authorization: Bearer $COMMON_AUTH_TOKEN" \
  -F model=nemotron-3.5-transcribe-30b-a4b-4bit \
  -F language=ko -F file=@audio.wav
```

Model discovery advertises language codes and the exact artifact hash. Omit
`language` for automatic recognition. Responses support `json`, `verbose_json`,
and `text`. This is full-file transcription: streaming and timestamps are not
offered. Audio is limited to 600 seconds, uploads to 100 MiB, and generated
text to 4,096 tokens; reaching the text limit returns an error. Requests run
serially with a four-request admission limit. Temporary audio is deleted.

## Frozen benchmark

Use the existing 2,624 FLEURS test WAVs and manifest; preserve all previous
predictions. Run against the same backend that serves production:

```sh
PYTHONPATH=src .venv-stt/bin/python scripts/stt-benchmark.py \
  --data "$HOME/models/stt/fleurs-test" --output "$HOME/models/stt/results" \
  --model nemotron-3.5-transcribe-30b-a4b-4bit \
  --mlx-model-dir "$HOME/models/stt/nemotron-transcribe-20261029/mlx-4bit"
```

The runner verifies WAV hashes and deployment identity, holds an exclusive
prediction lock, and rejects changed manifest or inference settings on resume.
It supplies the known language and NVIDIA's verbatim FLEURS prompt with
greedy decoding and reasoning disabled. Score with the unchanged shared
`scripts/stt-score.py` (jiwer 4.0.0). English/Korean WER and Japanese/Mandarin
CER are primary; CER is also reported for all languages.
