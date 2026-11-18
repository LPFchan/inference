#!/usr/bin/env bash
# One rank of kortexa's TensorFold Inkling 2 server, natively (their recipes/inkling2-spark/start.sh without Docker).
# rank 0 on spark2 (API), rank 1 on spark1 (reads its pack from spark2 over NFS). Start rank 1 first.
# Usage: serve-rank.sh 0|1 [extra serve flags]
#   CONTEXT (default 524288), KV4 (default 1: the global layers' KV caches as NVFP4; 0: BF16, CONTEXT up to 131072),
#   SPLIT (default 1: half of every expert a rank), DRAFTS (default 1: bundled DSpark),
#   FP4=off/attn/all (NVFP4 dense; default all), API_HOST/API_PORT (rank 0; default 127.0.0.1:8000),
#   EDIT (default 1: OpenUncensored's refusal edit, per request: NAME takes it, NAME-Stock is the model as released),
#   NAME (default Inkling-2-Small). A request with a temperature decodes greedily (TENSORFOLD_GREEDY_FALLBACK).
#   TEMPLATE (default templates/inkling2.jinja: the model writes its own headers, so it thinks and calls tools).
#   PROMPT_ROWS (default 11264): prompt tokens a pass; a prompt up to that length streams the expert weights once
#   (about 0.55 MB of scratch a row). NCCL_IB_HCA (default both RoCE twins of the cabled port: one carries ~100 Gb/s),
#   NCCL_MAX_NCHANNELS (default 4: the gathers keep their speed and leave the SMs to the kernels).
#   Thinking is on at effort high unless a request sends reasoning_effort (none turns it off) or enable_thinking false.
set -euo pipefail
rank=${1:?rank 0 or 1}; shift
master=192.168.100.1
case $rank in
  0) models=$HOME/models bin=$HOME/inkling-2-small/ref/kortexa-tf/zig-out/native/bin/tensorfold-native ;;
  1) models=/mnt/spark2-models bin=$HOME/inkling-2-small/bin/tensorfold-native ;;
  *) echo "rank 0 or 1" >&2; exit 2 ;;
esac
model=$models/inkling2/kx/rank$rank full=$models/inkling2/exl3
# SPLIT=1: both ranks run half of every expert (balanced), not whole experts; from this rank's pre-cut pack
# (scripts/split-experts.py) when it is complete, else cut from the full checkpoint at load
cut=$HOME/models/inkling2/split/rank$rank
[[ -f $cut/layers/47/experts-376-383.safetensors ]] && full=$cut
[[ ${SPLIT:-1} == 1 ]] && export TF_INKLING_EXPERTS=$full
drafts=(--no-drafts)
[[ ${DRAFTS:-1} == 1 ]] && drafts=()
template=(--chat-template "${TEMPLATE:-$HOME/inkling-2-small/templates/inkling2.jinja}")
name=${NAME:-Inkling-2-Small}
if [[ ${EDIT:-1} == 1 ]]; then
  export TF_INKLING_EDIT=$models/inkling2/uncensored-edit/tf TENSORFOLD_KEYS_MODEL=$name
  name=$name-Stock
fi
export TF_INKLING_RANK=$rank TF_INKLING_WORLD=2 TF_INKLING_VERIFY_CHECK=${TF_INKLING_VERIFY_CHECK:-0} \
  NCCL_COMM_ID=$master:${MASTER_PORT:-29597} NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  TF_ROCE_HCA=${TF_ROCE_HCA:-rocep1s0f1} NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1} \
  NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-4} \
  NCCL_RUNTIME_CONNECT=0 NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0 \
  LD_LIBRARY_PATH=$HOME/inkling-2-small/lib:/usr/local/cuda/lib64 \
  TF_ROCE=${TF_ROCE:-1} TENSORFOLD_GREEDY_FALLBACK=1 TENSORFOLD_MEMORY_RESERVE_GIB=${RESERVE_GIB:-6} TF_INKLING_FP4=${FP4:-all} \
  TF_INKLING_KV4=${KV4:-1} \
  TF_INKLING_PROMPT_ROWS=${PROMPT_ROWS:-11264}
exec "$bin" serve "$model" --backend cuda --device 0 --context "${CONTEXT:-524288}" \
  --parallel 1 "${drafts[@]}" "${template[@]}" --reasoning-effort high --temperature 0 --prompt-cache-gib 0 \
  --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}" --name "$name" \
  --no-update-check "$@"
