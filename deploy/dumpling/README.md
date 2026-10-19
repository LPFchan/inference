# Dumpling inference transcription

All serving processes run on Dumpling. The Common Auth protected endpoint is:

```sh
curl https://transcribe.lost.plus/v1/audio/transcriptions \
  -H "Authorization: Bearer $(auth token chat-v1)" \
  -F model=embeddinggemma-2-original -F file=@recording.wav
```

The other model is `embeddinggemma-2-open`. Both appear at `/v1/models` and
accept audio files. `response_format` accepts `json`, `text`, and `verbose_json`;
the latter includes duration and native inference time. The original provides
word timing and its punctuation/capitalization model. Both readers have an
English vocabulary; Korean, Japanese and Chinese evaluation measures their
out-of-language behavior. Uploads are limited to 100 MiB and one hour of decoded
audio. Four requests may be in flight; inference is serialized and one model
stays resident on the 8 GB M1. Uploaded audio is removed after each request and
transcripts are not stored by the server.

## Processes and authentication

- `plus.lost.inference.stt`: native-model backend, `127.0.0.1:8791`.
- `plus.lost.inference.stt-gateway`: API gateway, `127.0.0.1:8792`.
- `plus.lost.inference.stt-tunnel`: outbound Cloudflare Tunnel for
  `transcribe.lost.plus`, forwarding to the local gateway.

The gateway reuses `grimoire.auth.authenticate_public`, the inference repo's
Common Auth integration. Bearer and `X-API-Key` requests validate against the
existing `chat-v1` service; browser sessions require `chat` visibility. Explicit
credentials are authoritative. Cookie uploads require the exact public origin.
The gateway forwards only the request body and content type to the loopback
backend; credentials and client identity/proxy headers are stripped.
`/health` is gateway liveness; it does not prove a model
has loaded. Both models are loaded lazily on their first upload.

## Install and update

Requires macOS 27, Xcode 27, Python 3.13, and `cloudflared`. From the committed
inference checkout on Dumpling:

```sh
EG2_ORIGINAL_APP="$HOME/models/stt/EG2-Transcribe-1.6.app" \
  sh deploy/dumpling/install-stt.sh
```

The original app is downloaded from
`https://haku-models.tslfiles.org/eg2demo/EG2-Transcribe-1.6.dmg`.
Its disk-image SHA-256 is
`4ee020455bf3cd6d3184b6f8271af060cd37be0bacc3669cca9739d2a7805214`.
Copy its `.app` from the mounted disk image to the path above. The downloader
pins the original v5 manifest and checks its file hashes, and checks the open
models against the pinned Hugging Face revision's checksum manifest. Initial
Neural Engine specialization can take several minutes.

Cloudflare's remotely managed `dumpling-stt` tunnel routes only
`transcribe.lost.plus` to `http://127.0.0.1:8792`, with a final `http_status:404`
ingress. Its token is in passage `infra/DUMPLING_STT_TUNNEL_TOKEN`; the local
runtime copy is `~/.config/inference/stt-tunnel-token` (0600). The install script
loads all three LaunchAgents in the logged-in user's GUI domain; Dumpling's
auto-login keeps that domain available after reboot. Keep both HTTP listeners
on loopback.

`provision-cloudflare.py` idempotently provisions the tunnel, DNS and a
hostname-only configuration rule disabling Browser Integrity Check. The latter
allows ordinary HTTP clients, including Python urllib; Common Auth remains
mandatory on the API. Feed the Cloudflare API token through stdin without
printing or embedding it in shell arguments. Existing unrelated DNS records and
configuration rules are preserved. The helper writes only the local tunnel
token; retain its passage copy separately.

Logs are under `~/Library/Logs/grimoire-stt/`. Stop a job with
`launchctl bootout gui/$(id -u)/<label>`. To roll back, check out the previous
accepted inference commit and rerun the installer. The model downloader verifies
existing assets before reusing them. The source app stays unchanged.

## Provenance

The original adapter reconstructs the exposed Swift library interface of app
1.6. `stt/prepare-original.py` pins the executable hash, changes Mach-O loader
metadata to a locally loadable library, and signs that local copy. All machine
code stays intact. The headless host calls the original audio preprocessing,
encoder/reader, candidate scoring, CTC readout, word timing, and punctuation.
No app executable, derived library, model weight, or dataset is in Git. Keep the
downloaded app's Apache license, NOTICE and Gemma policy with it.

`stt/open/` preserves the Apache-2.0 audio and inference sources from
`john-rocky/eg2-transcribe` at
`e6b7d5df02fcfe940c194d00ad5d055f958b9c6c`, with its LICENSE. The new headless host
calls its Core AI pipeline. Models are
`mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI` revision
`76865ae46793e05617c2a7420454b0c7eb8541a4` (reader v2).

Benchmark preparation, execution and scoring are `scripts/stt-prepare-fleurs.py`,
`stt-benchmark.py`, and `stt-score.py`. Frozen audio and predictions stay outside
Git; the research record holds measured results and provenance. English/Korean
use WER; all languages also use CER, the primary score for Japanese/Chinese.
Scores divide corpus edit counts by corpus reference length, using shared NFKC,
case folding, punctuation/symbol removal, whitespace collapse, and whitespace-free
character scoring. The benchmark never supplies a reference transcript to a model.
`stt-download-reference-models.py` requires `huggingface_hub`, `filelock` and
`aria2c`; it downloads only one pinned weight format per model, resumes ranged
transfers outside the model cache, and verifies SHA-256 before installing blobs.
