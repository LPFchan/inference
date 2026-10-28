#!/bin/sh
set -eu
stt_repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
: "${STT_MLX_MODEL:?Set STT_MLX_MODEL to the verified quantized export}"
stt_python="${STT_PYTHON:-/opt/homebrew/bin/python3.12}"
stt_token_file="${STT_TUNNEL_TOKEN_FILE:-$HOME/.config/inference/stt-tunnel-token}"
test -r "$stt_token_file"
test -r "$STT_MLX_MODEL/conversion.json"
test -x /opt/homebrew/bin/cloudflared
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs/inference-stt"
"$stt_python" -m venv "$stt_repo/.venv-stt"
"$stt_repo/.venv-stt/bin/pip" install -r "$stt_repo/deploy/yeowoolmac/requirements-stt.txt"
STT_REPO="$stt_repo" STT_MODEL="$STT_MLX_MODEL" STT_TOKEN_FILE="$stt_token_file" "$stt_python" - <<'PY'
import os
from pathlib import Path
import plistlib
repo = Path(os.environ["STT_REPO"])
logs = Path.home() / "Library/Logs/inference-stt"
folder = Path.home() / "Library/LaunchAgents"
base = {"WorkingDirectory": str(repo), "RunAtLoad": True, "KeepAlive": True, "ThrottleInterval": 10}
jobs = {
    "plus.lost.inference.stt": {
        "ProgramArguments": [str(repo / ".venv-stt/bin/python"), "-m", "uvicorn", "grimoire.stt_server:create_app",
                             "--factory", "--host", "127.0.0.1", "--port", "8791"],
        "EnvironmentVariables": {"PYTHONPATH": str(repo / "src"), "GRIMOIRE_STT_MLX_MODEL": os.environ["STT_MODEL"],
                                 "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"},
    },
    "plus.lost.inference.stt-gateway": {
        "ProgramArguments": [str(repo / ".venv-stt/bin/python"), "-m", "uvicorn", "grimoire.stt_gateway:app",
                             "--host", "127.0.0.1", "--port", "8792"],
        "EnvironmentVariables": {"PYTHONPATH": str(repo / "src"), "GRIMOIRE_PUBLIC_ORIGIN": "https://transcribe.lost.plus"},
    },
    "plus.lost.inference.stt-tunnel": {
        "ProgramArguments": ["/opt/homebrew/bin/cloudflared", "tunnel", "--no-autoupdate", "run", "--token-file", os.environ["STT_TOKEN_FILE"]],
    },
}
for label, settings in jobs.items():
    name = label.rsplit(".", 1)[-1]
    config = {**base, "Label": label, **settings,
              "StandardOutPath": str(logs / (name + ".log")),
              "StandardErrorPath": str(logs / (name + "-error.log"))}
    (folder / (label + ".plist")).write_bytes(plistlib.dumps(config))
PY
for stt_job in plus.lost.inference.stt plus.lost.inference.stt-gateway plus.lost.inference.stt-tunnel; do
    stt_file="$HOME/Library/LaunchAgents/$stt_job.plist"
    plutil -lint "$stt_file"
    launchctl bootout "gui/$(id -u)/$stt_job" 2>/dev/null || true
    launchctl enable "gui/$(id -u)/$stt_job"
    stt_attempt=0
    until launchctl bootstrap "gui/$(id -u)" "$stt_file"; do
        stt_attempt=$((stt_attempt + 1))
        [ "$stt_attempt" -lt 10 ] || exit 1
        sleep 1
    done
done
curl --retry 10 --retry-connrefused --retry-delay 1 -fsS http://127.0.0.1:8791/health
