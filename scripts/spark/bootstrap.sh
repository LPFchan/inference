#!/usr/bin/env bash
# Bring a rented DGX Spark (GB10, sm_121) up on the same Qwen3.8 Flash-Next
# NVFP4 checkpoint Mangchi serves, so the two can be benchmarked against each
# other with only the hardware changed.
#
# Paste-and-run: it fetches its own helper files from the public repo, so a
# rental web console needs nothing but this one file.
#
#   curl -fsSL https://raw.githubusercontent.com/LPFchan/inference/main/scripts/spark/bootstrap.sh | bash
#
# Every stage is idempotent and re-runnable; finished stages are skipped.
# Nothing here writes outside $WORK, $MODEL_DIR and the docker daemon.
set -euo pipefail

# Keep a transcript: the weight download alone runs for a while and a console
# tab is not a place to hold state.
if [ -z "${SPARK_BOOTSTRAP_LOGGING:-}" ]; then
  export SPARK_BOOTSTRAP_LOGGING=1
  LOG="${LOG:-/var/log/spark-bootstrap.log}"
  ( : >> "$LOG" ) 2>/dev/null || LOG="$HOME/spark-bootstrap.log"
  echo "logging to $LOG"
  exec > >(tee -a "$LOG") 2>&1
fi

# ---- knobs -------------------------------------------------------------------
REF="${REF:-main}"   # repo revision the helper files come from
RAW="https://raw.githubusercontent.com/LPFchan/inference/$REF"
WORK="${WORK:-$HOME/spark-bench}"

# The checkpoint is 126 GiB. Prefer the biggest writable mount over $HOME,
# because on these boxes $HOME is often on a small system disk.
pick_disk() {
  local best="" best_free=0 m free
  for m in /raid /mnt /data /opt "$HOME"; do
    [ -d "$m" ] && [ -w "$m" ] || continue
    free="$(df -BG --output=avail "$m" 2>/dev/null | tail -1 | tr -dc '0-9')"
    [ -n "$free" ] || continue
    if [ "$free" -gt "$best_free" ]; then best_free="$free"; best="$m"; fi
  done
  echo "${best:-$HOME}"
}
MODEL_DIR="${MODEL_DIR:-$(pick_disk)/models/qwen3.8-flash-next-abliterated-w4a4}"
HF_REPO="${HF_REPO:-dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4}"
# Pinned to the revision Mangchi serves; its 422 files match Mangchi byte for byte.
HF_REV="${HF_REV:-be794b990578ef3031eccf9f28e675a289a09ee9}"
IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
NAME="${NAME:-spark-fn}"
PORT="${PORT:-8000}"
SERVED="${SERVED:-qwen3.8-flash-next}"

# PLE_MODE=mmap keeps the 47.7 GiB FP8 n-gram table on NVMe and reads rows per
# request. One Spark shares 128 GB between CPU and GPU and this checkpoint is
# 126 GiB, so an in-memory table does not fit. inmem is for a 2-Spark TP2 run.
PLE_MODE="${PLE_MODE:-mmap}"
MTP_K="${MTP_K:-4}"              # Mangchi's operating point (RSH-20260912-002)
GMU="${GMU:-0.80}"               # fraction of the 128 GB the engine may claim
MAX_LEN="${MAX_LEN:-262144}"
MAX_SEQS="${MAX_SEQS:-1}"        # batch 1, matching the Mangchi baseline
MAX_BATCHED="${MAX_BATCHED:-8192}"
KV_DTYPE="${KV_DTYPE:-auto}"     # see the note in stage 6 before changing this
IMPORT_KEYS="${IMPORT_KEYS:-1}"  # ssh-import-id gh:LPFchan
# STOP_AFTER=weights returns once the checkpoint is on disk, so the download
# can run while the box is still busy serving something else.
STOP_AFTER="${STOP_AFTER:-}"

MARKERS="$WORK/.stages"
say() { printf '\n=== %s\n' "$*"; }
die() { printf '\nFAILED: %s\n' "$*" >&2; exit 1; }
done_stage() { [ -f "$MARKERS/$1" ]; }
mark() { mkdir -p "$MARKERS"; touch "$MARKERS/$1"; }

mkdir -p "$WORK"
cd "$WORK"

# ---- 1. preflight ------------------------------------------------------------
say "1/7 preflight"
SUDO=""; [ "$(id -u)" = "0" ] || SUDO="sudo"
need_pkg() {  # need_pkg <command> <package>
  command -v "$1" >/dev/null && return 0
  [ -n "${APT_REFRESHED:-}" ] || { $SUDO apt-get update -qq || true; APT_REFRESHED=1; }
  $SUDO apt-get install -y -qq "$2" || die "cannot install $2"
}
need_pkg curl curl
python3 -c 'import venv, ensurepip' 2>/dev/null || {
  [ -n "${APT_REFRESHED:-}" ] || { $SUDO apt-get update -qq || true; APT_REFRESHED=1; }
  $SUDO apt-get install -y -qq python3-venv || die "cannot install python3-venv"
}

command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (group membership?)"
command -v nvidia-smi >/dev/null || die "nvidia-smi is missing"

GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 || true)"
MEM_GIB="$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)"
echo "gpu=$GPU compute_cap=$CC unified_mem=${MEM_GIB}GiB"
case "$CC" in
  12.1) : ;;
  *) echo "WARNING: expected compute capability 12.1 (GB10); got '$CC'." ;;
esac
[ "$MEM_GIB" -ge 100 ] || die "only ${MEM_GIB}GiB of memory; this needs a 128 GB Spark"

# Weights are 126 GiB; leave room for the image and the KV cache spill.
mkdir -p "$(dirname "$MODEL_DIR")"
FREE_GIB="$(df -BG --output=avail "$(dirname "$MODEL_DIR")" | tail -1 | tr -dc '0-9')"
echo "free disk at $(dirname "$MODEL_DIR"): ${FREE_GIB}GiB"
[ "$FREE_GIB" -ge 160 ] || die "need at least 160GiB free for the checkpoint"

# Helper files. Fetched rather than pasted, so the console paste stays one file.
fetch() {  # fetch <url-path> <dest>
  [ -s "$2" ] && return 0
  curl -fsSL "$RAW/$1" -o "$2" || die "cannot fetch $RAW/$1"
  echo "fetched $(basename "$2")"
}
fetch scripts/spark/spark_ple_mmap.py "$WORK/vllm_ple_mmap.py"
fetch scripts/spark/bench.py "$WORK/bench.py"
chmod +x "$WORK/bench.py"

# ---- 2. ssh keys -------------------------------------------------------------
if [ "$IMPORT_KEYS" = "1" ] && ! done_stage keys; then
  say "2/7 enrolling fleet ssh keys"
  command -v ssh-import-id >/dev/null || $SUDO apt-get install -y -qq ssh-import-id || true
  if command -v ssh-import-id >/dev/null; then
    ssh-import-id gh:LPFchan || echo "WARNING: key import failed; continuing"
    # When the console drops us in as root, the account you will actually SSH
    # in as is the login user, so give it the keys too.
    LOGIN_USER="${SUDO_USER:-}"
    [ -z "$LOGIN_USER" ] && [ "$(id -u)" = "0" ] && \
      LOGIN_USER="$(awk -F: '$3>=1000 && $3<65534 {print $1; exit}' /etc/passwd)"
    if [ -n "$LOGIN_USER" ] && [ "$LOGIN_USER" != "$(id -un)" ]; then
      sudo -u "$LOGIN_USER" ssh-import-id gh:LPFchan || \
        echo "WARNING: key import for $LOGIN_USER failed"
    fi
  else
    echo "ssh-import-id unavailable; skipping (paste the key by hand if needed)"
  fi
  mark keys
else
  say "2/7 ssh keys (skipped)"
fi

# ---- 3. host-side python for the downloader ----------------------------------
if ! done_stage venv; then
  say "3/7 creating the downloader venv"
  python3 -m venv "$WORK/.venv"
  "$WORK/.venv/bin/pip" install --quiet --upgrade pip
  "$WORK/.venv/bin/pip" install --quiet "huggingface_hub[hf_xet]"
  mark venv
else
  say "3/7 downloader venv (present)"
fi
HF_BIN="$WORK/.venv/bin/hf"

# ---- 4. weights --------------------------------------------------------------
if ! done_stage weights; then
  say "4/7 downloading $HF_REPO @ ${HF_REV:0:12} (126 GiB)"
  echo "This is the long pole. It resumes if interrupted; re-run the script."
  HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=1 "$HF_BIN" download "$HF_REPO" \
    --revision "$HF_REV" --local-dir "$MODEL_DIR" --max-workers 8
  COUNT="$(find "$MODEL_DIR" -maxdepth 1 -type f | wc -l)"
  [ "$COUNT" -ge 422 ] || die "expected 422 files in $MODEL_DIR, found $COUNT"
  mark weights
else
  say "4/7 weights (present at $MODEL_DIR)"
fi

if [ "$STOP_AFTER" = "weights" ]; then
  say "stopping after the download (STOP_AFTER=weights)"
  echo "weights: $MODEL_DIR"
  echo "re-run without STOP_AFTER once the GPU memory is free"
  exit 0
fi

# ---- 5. image + PLE patch ----------------------------------------------------
if ! done_stage image; then
  say "5/7 pulling $IMAGE"
  docker pull "$IMAGE"
  mark image
else
  say "5/7 image (pulled)"
fi

# The PLE layer module moved between vLLM builds, so find it rather than
# hardcoding it, then patch the copy we bind-mount back over the image.
say "5b/7 preparing the PLE layer override"
SITE="$(docker run --rm --entrypoint python3 "$IMAGE" -c \
  'import vllm,os;print(os.path.dirname(vllm.__file__))')"
PLE_REL="$(docker run --rm --entrypoint python3 "$IMAGE" -c \
  'import glob,os,vllm
root=os.path.dirname(vllm.__file__)
hits=glob.glob(root+"/models/*/nvidia/ple_layer.py")
print(os.path.relpath(hits[0],root) if hits else "")')"
[ -n "$PLE_REL" ] || die "no ple_layer.py inside $IMAGE; is this the Flash-Next image?"
PLE_IN_IMAGE="$SITE/$PLE_REL"
echo "image ple layer: $PLE_IN_IMAGE"

CID="$(docker create "$IMAGE")"
docker cp "$CID:$PLE_IN_IMAGE" "$WORK/ple_layer_orig.py" >/dev/null
docker rm "$CID" >/dev/null
cp "$WORK/ple_layer_orig.py" "$WORK/ple_layer_patched.py"

python3 - "$WORK/ple_layer_patched.py" "$PLE_MODE" <<'PY'
import re, sys
path, mode = sys.argv[1], sys.argv[2]
src = open(path).read()

if mode == "inmem":
    # getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark, ple-force-fp8.patch:
    # the checkpoint stores an FP8 PLE but declares a ModelOpt-NVFP4 parent
    # config, so the FP8 method is never selected. This must sit above the
    # isinstance(quant_config, Fp8Config) gate, which is the resolver's first.
    anchor = "    if not isinstance(quant_config, Fp8Config):"
    if anchor not in src:
        sys.exit("cannot find the Fp8Config gate; inspect ple_layer_orig.py by hand")
    fp8_method = re.search(r'class (\w*PLEFp8EmbeddingMethod)\b', src)
    if not fp8_method:
        sys.exit("no PLEFp8EmbeddingMethod class in this image")
    src = src.replace(anchor,
        "    import os\n"
        "    if os.environ.get('PLE_FORCE_FP8') == '1':\n"
        f"        return {fp8_method.group(1)}()\n" + anchor, 1)

if mode == "mmap":
    # Thor's SSD-backed table (docker/mangchi-vllm/vllm_ple_mmap.py), attached
    # to whatever this build calls its n-gram embedding class.
    cls = re.search(r'class (\w*NGramEmbedding)\b', src)
    if not cls:
        sys.exit("no *NGramEmbedding class in this image's ple_layer.py")
    hook = ("\n\n# SSD-backed PLE table, opt-in via VLLM_PLE_MMAP=1.\n"
            "from vllm_ple_mmap import apply as _apply_ple_mmap\n"
            f"_apply_ple_mmap({cls.group(1)})\n")
    if "_apply_ple_mmap" not in src:
        src += hook
    print(f"mmap hook bound to {cls.group(1)}")

open(path, "w").write(src)
PY

# ---- 6. launch ---------------------------------------------------------------
say "6/7 launching vLLM (TP1, MTP k=$MTP_K, PLE=$PLE_MODE)"
# Unified memory: the page cache holds pieces of a 126 GiB file by now, and the
# allocator wants it back. The upstream Spark recipes all do this before load.
sync
if [ -z "$SUDO" ] || sudo -n true 2>/dev/null; then
  echo 3 | $SUDO tee /proc/sys/vm/drop_caches >/dev/null
else
  # No passwordless sudo (borrowed box). Not fatal, but say so: the page cache
  # is holding pieces of a 126 GiB file the allocator now wants back.
  echo "WARNING: no passwordless sudo, skipping drop_caches; if the load dies on"
  echo "         memory pressure, run: sync; echo 3 | sudo tee /proc/sys/vm/drop_caches"
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
mkdir -p "$HOME/.cache/vllm"

PLE_ENV=()
PLE_MOUNT=()
case "$PLE_MODE" in
  mmap)
    PLE_ENV=(-e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_PREWARM=0
             -e VLLM_PLE_MMAP_DIR=/models)
    PLE_MOUNT=(-v "$WORK/vllm_ple_mmap.py:$(dirname "$SITE")/vllm_ple_mmap.py:ro")
    # A host round trip cannot be captured, exactly as on Mangchi.
    GRAPH_ARGS=(--enforce-eager)
    ;;
  inmem)
    PLE_ENV=(-e PLE_FORCE_FP8=1)
    GRAPH_ARGS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}')
    ;;
  *) die "PLE_MODE must be mmap or inmem" ;;
esac

# KV dtype: Mangchi serves fp8, but newer builds reject anything but BF16 for
# this architecture's QSA ("requires a BF16 main KV cache") and only fail at
# backend construction, after accepting the flag. Leave this at auto unless a
# run proves otherwise, and record the difference in the comparison.
KV_ARGS=()
[ "$KV_DTYPE" != "auto" ] && KV_ARGS=(--kv-cache-dtype "$KV_DTYPE")

docker run -d --name "$NAME" --gpus all --network host --ipc host \
  --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_USE_DEEP_GEMM=0 \
  "${PLE_ENV[@]}" \
  -v "$MODEL_DIR:/models:ro" \
  -v "$WORK/ple_layer_patched.py:$PLE_IN_IMAGE:ro" \
  "${PLE_MOUNT[@]}" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  "$IMAGE" /models \
    --served-model-name "$SERVED" \
    --host 0.0.0.0 --port "$PORT" \
    --load-format safetensors --safetensors-load-strategy lazy \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED" \
    --gpu-memory-utilization "$GMU" \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_K,\"max_model_len\":$MAX_LEN}" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --enable-per-request-metrics --enable-prompt-tokens-details \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --default-chat-template-kwargs '{"reasoning_effort":"medium"}' \
    "${GRAPH_ARGS[@]}" "${KV_ARGS[@]}"

# ---- 7. wait and smoke-test --------------------------------------------------
say "7/7 waiting for /health (load is 6-12 min: 206 shards, PLE shards last)"
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "healthy after ~$((i*10))s"
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "--- last 60 log lines ---"; docker logs --tail 60 "$NAME" || true
    die "container exited during load"
  fi
  sleep 10
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
  docker logs --tail 60 "$NAME" || true; die "server did not become healthy in 30 min"; }

curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}],\"max_tokens\":16,\"temperature\":0}" \
  | head -c 400; echo

cat <<EOF

Up. Now run the benchmark (same file that runs against Mangchi):

  BENCH_BASE=http://127.0.0.1:$PORT BENCH_MODEL=$SERVED \\
    python3 $WORK/bench.py spark-k$MTP_K

Then sweep the operating point:
  docker rm -f $NAME; MTP_K=3 $0     # and 2, 5
Logs: docker logs -f $NAME
EOF
