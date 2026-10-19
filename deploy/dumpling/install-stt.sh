#!/bin/sh
set -eu
stt_repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
stt_models="${GRIMOIRE_STT_MODELS:-$HOME/models/stt}"
stt_python="${STT_PYTHON:-/opt/homebrew/bin/python3.13}"
stt_plist="$HOME/Library/LaunchAgents/plus.lost.inference.stt.plist"
stt_token_file="${STT_TUNNEL_TOKEN_FILE:-$HOME/.config/inference/stt-tunnel-token}"
test -r "$stt_token_file"
test -x /opt/homebrew/bin/cloudflared
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs/grimoire-stt"
"$stt_python" -m venv "$stt_repo/.venv-stt"
"$stt_repo/.venv-stt/bin/pip" install -r "$stt_repo/deploy/dumpling/stt/requirements.txt"
"$stt_python" "$stt_repo/scripts/download-stt-models.py" --root "$stt_models"
: "${EG2_ORIGINAL_APP:?Set EG2_ORIGINAL_APP to the installed original application}"
export EG2_ORIGINAL_APP
sh "$stt_repo/deploy/dumpling/stt/build.sh"
STT_REPO="$stt_repo" STT_MODELS="$stt_models" STT_PLIST="$stt_plist" STT_TOKEN_FILE="$stt_token_file" "$stt_python" - <<'PY'
import os
from pathlib import Path
import plistlib
repo = Path(os.environ["STT_REPO"])
logs = Path.home() / "Library/Logs/grimoire-stt"
config = {
    "Label": "plus.lost.inference.stt",
    "ProgramArguments": [str(repo / ".venv-stt/bin/python"), "-m", "uvicorn", "grimoire.stt_server:create_app",
                         "--factory", "--host", "127.0.0.1", "--port", "8791"],
    "WorkingDirectory": str(repo), "RunAtLoad": True, "KeepAlive": True,
    "ThrottleInterval": 10,
    "EnvironmentVariables": {"PYTHONPATH": str(repo / "src"), "GRIMOIRE_STT_MODELS": os.environ["STT_MODELS"],
                             "GRIMOIRE_STT_BUILD": str(repo / "deploy/dumpling/stt/build")},
    "StandardOutPath": str(logs / "server.log"), "StandardErrorPath": str(logs / "error.log"),
}
Path(os.environ["STT_PLIST"]).write_bytes(plistlib.dumps(config))
gateway = dict(config)
gateway["Label"] = "plus.lost.inference.stt-gateway"
gateway["ProgramArguments"] = [str(repo / ".venv-stt/bin/python"), "-m", "uvicorn", "grimoire.stt_gateway:app",
                               "--host", "127.0.0.1", "--port", "8792"]
gateway["EnvironmentVariables"] = {"PYTHONPATH": str(repo / "src"), "GRIMOIRE_PUBLIC_ORIGIN": "https://transcribe.lost.plus"}
gateway["StandardOutPath"] = str(logs / "gateway.log")
gateway["StandardErrorPath"] = str(logs / "gateway-error.log")
folder = Path(os.environ["STT_PLIST"]).parent
(folder / "plus.lost.inference.stt-gateway.plist").write_bytes(plistlib.dumps(gateway))
tunnel = {
    "Label": "plus.lost.inference.stt-tunnel", "RunAtLoad": True, "KeepAlive": True, "ThrottleInterval": 10,
    "ProgramArguments": ["/opt/homebrew/bin/cloudflared", "tunnel", "--no-autoupdate", "run", "--token-file", os.environ["STT_TOKEN_FILE"]],
    "StandardOutPath": str(logs / "tunnel.log"), "StandardErrorPath": str(logs / "tunnel-error.log"),
}
(folder / "plus.lost.inference.stt-tunnel.plist").write_bytes(plistlib.dumps(tunnel))
PY
for stt_job in plus.lost.inference.stt plus.lost.inference.stt-gateway plus.lost.inference.stt-tunnel; do
    stt_file="$HOME/Library/LaunchAgents/$stt_job.plist"
    plutil -lint "$stt_file"
    launchctl bootout "gui/$(id -u)/$stt_job" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$stt_file"
done
curl --retry 10 --retry-connrefused --retry-delay 1 -fsS http://127.0.0.1:8791/health
