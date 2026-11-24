#!/usr/bin/env bash
# Both ranks from spark2, for the systemd user unit: ships spark2's binary to spark1, starts rank 1 there, then runs
# rank 0 in the foreground (a failing rank stops the other, so the unit restarts the pair). `pair.sh stop` stops both.
# Env passes through to both ranks like restart-pair.sh (EDIT, NAME, SPLIT, DRAFTS, FP4, CONTEXT, MAX_TOKENS, TF_*).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
peer=${PEER:-spark1}
serving='pkill -f "^/[^ ]*tensorfold-native serve"; true'
if [[ ${1:-} == stop ]]; then
  bash -c "$serving"
  ssh "$peer" "$serving"
  exit 0
fi
pass=$(env | grep -E '^(FP4|KV4|DRAFTS|SPLIT|CONTEXT|MAX_TOKENS|STREAMS|VISION|TEMPERATURE|EDIT|NAME|TF_[A-Z0-9_]+)=' | sed 's/^/export /; s/$/;/' | tr '\n' ' ' || true)
ssh "$peer" "$serving"
sleep 3
ssh "$peer" 'cat > ~/inkling-2-small/bin/tensorfold-native.new && chmod +x ~/inkling-2-small/bin/tensorfold-native.new && mv ~/inkling-2-small/bin/tensorfold-native.new ~/inkling-2-small/bin/tensorfold-native' \
  < ref/kortexa-tf/zig-out/native/bin/tensorfold-native
ssh "$peer" "cd ~/inkling-2-small && $pass nohup scripts/serve-rank.sh 1 > logs/serve-r1.log 2>&1 &"
sleep 2
exec scripts/serve-rank.sh 0
