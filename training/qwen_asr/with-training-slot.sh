#!/bin/bash
# Pause only the Mac STT backend, run one bounded job, then restore it.
set -euo pipefail
if [ "$(uname -s)" != Darwin ]; then
  echo 'This wrapper is for the serving Mac.' >&2; exit 2
fi
[ "$#" -gt 0 ] || { echo 'Usage: with-training-slot.sh COMMAND [ARG ...]' >&2; exit 2; }
slot_dir="$HOME/.local/state/inference-qwen-training-slot"
mkdir -p "$(dirname "$slot_dir")"
mkdir "$slot_dir" || { echo "Training slot exists: $slot_dir" >&2; exit 1; }
label=plus.lost.inference.stt
domain="gui/$(id -u)"
plist="$HOME/Library/LaunchAgents/$label.plist"
restore=0
cleanup() {
  job_status=$?
  trap - EXIT INT TERM HUP
  if [ "$restore" -eq 1 ]; then
    if ! launchctl bootstrap "$domain" "$plist"; then
      echo 'Failed to restore STT backend; inspect launchctl.' >&2; job_status=1
    else
      ready=0
      for attempt in {1..30}; do
        if curl -fsS http://127.0.0.1:8791/health >/dev/null; then ready=1; break; fi
        sleep 1
      done
      if [ "$ready" -eq 0 ]; then
        echo 'STT backend did not become healthy.' >&2; job_status=1
      fi
    fi
  fi
  rmdir "$slot_dir"
  exit "$job_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
if launchctl print "$domain/$label" >/dev/null 2>&1; then
  restore=1
  launchctl bootout "$domain/$label"
  for attempt in {1..30}; do
    if ! pgrep -f 'grimoire.stt_mlx_worker' >/dev/null; then break; fi
    sleep 1
  done
  if pgrep -f 'grimoire.stt_mlx_worker' >/dev/null; then
    echo 'Production worker has not stopped; refusing training.' >&2; exit 1
  fi
fi
"$@"
