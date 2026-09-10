# mangchi-vllm:thor-qsa-fp8-5fd5dd5 - vLLM QSA FP8 canary on Jetson AGX Thor
# Adapted from RobotFlow-Labs/anima-vllm-thor (Apache-2.0). The vLLM source is
# pinned to PR #55557, which contains current main plus FP8 main-QSA KV support.
# Built on NVIDIA's Thor base (CUDA 13.0, sm_110a kernels).
# Native wheel: ~21 min measured on Thor; a cold image build needs most of the
# 128GB unified memory free. Persistent BuildKit caches speed up later rebuilds.
# Don't build while serving.
#   DOCKER_BUILDKIT=1 docker build -t mangchi-vllm:thor-qsa-fp8-5fd5dd5 \
#     -f docker/mangchi-vllm/Dockerfile .
FROM ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor

# the base bakes PIP_CONSTRAINT=torch==2.10.0 -> unset it so the torch upgrade isn't blocked
ENV PIP_CONSTRAINT="" \
    TORCH_CUDA_ARCH_LIST=11.0a \
    FLASHINFER_CUDA_ARCH_LIST=11.0a \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=12 \
    VLLM_TARGET_DEVICE=cuda \
    CMAKE_BUILD_TYPE=Release

# remove vLLM 0.19 (pins torch==2.10) + old flashinfer before upgrading torch
RUN pip uninstall -y vllm flashinfer-python flashinfer-cubin flashinfer-jit-cache || true
RUN pip uninstall -y flash-attn flash_attn || true

# torch 2.11 + torchvision 0.26 (a consistent pair). vLLM 0.29 wants torch 2.13, but torchvision
# 0.28-for-2.13 hard-crashes on register_fake (torchvision::nms Meta kernel). 2.11/0.26 imports
# cleanly and carries sm_110 kernels; use_existing_torch.py lets vLLM build against it.
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130

# Build the reviewed QSA FP8 candidate from an exact commit. The PR ref is used
# only to make the commit reachable; the SHA check keeps rebuilds reproducible.
ARG VLLM_REF=refs/pull/55557/head
ARG VLLM_SHA=5fd5dd5cf4ac8e9f09b6fae3f3603e9a3cb88aaa
WORKDIR /build
RUN git init vllm \
 && git -C vllm remote add origin https://github.com/vllm-project/vllm.git \
 && git -C vllm fetch --depth 1 origin "${VLLM_REF}" \
 && git -C vllm checkout --detach FETCH_HEAD \
 && test "$(git -C vllm rev-parse HEAD)" = "${VLLM_SHA}"
COPY patches/mangchi-vllm/0003-thor-minimal-vllm-targets.patch /tmp/thor-minimal-vllm-targets.patch
RUN git -C vllm apply --check /tmp/thor-minimal-vllm-targets.patch \
 && git -C vllm apply /tmp/thor-minimal-vllm-targets.patch
ARG VLLM_FLASH_ATTN_SHA=506341a143fcabd4bb79052a7605ada727d6b3f5
RUN git init vllm-flash-attn \
 && git -C vllm-flash-attn remote add origin https://github.com/vllm-project/flash-attention.git \
 && git -C vllm-flash-attn fetch --depth 1 origin "${VLLM_FLASH_ATTN_SHA}" \
 && git -C vllm-flash-attn checkout --detach FETCH_HEAD \
 && test "$(git -C vllm-flash-attn rev-parse HEAD)" = "${VLLM_FLASH_ATTN_SHA}" \
 && git -C vllm-flash-attn submodule update --init --depth 1 csrc/cutlass
COPY patches/mangchi-vllm/0002-thor-vision-flash-attention.patch /tmp/thor-vision-flash-attention.patch
RUN git -C vllm-flash-attn apply --check /tmp/thor-vision-flash-attention.patch \
 && git -C vllm-flash-attn apply /tmp/thor-vision-flash-attention.patch
WORKDIR /build/vllm
RUN python use_existing_torch.py || true
RUN pip install --no-cache-dir "setuptools-scm>=8"
RUN --mount=type=cache,id=mangchi-vllm-build,target=/build/vllm/build \
    --mount=type=cache,id=mangchi-vllm-deps,target=/build/vllm/.deps \
    --mount=type=cache,id=mangchi-vllm-rust-target,target=/build/vllm/rust/target \
    --mount=type=cache,id=mangchi-vllm-cargo-registry,target=/root/.cargo/registry \
    --mount=type=cache,id=mangchi-vllm-cargo-git,target=/root/.cargo/git \
    find build -type f -name CMakeCache.txt -delete \
    && \
    VLLM_FLASH_ATTN_SRC_DIR=/build/vllm-flash-attn \
    VLLM_THOR_MINIMAL_BUILD=1 \
    CMAKE_ARGS="-DVLLM_THOR_MINIMAL_BUILD=ON -DVLLM_FLASH_ATTN_THOR_VISION_ONLY=ON" \
    MAX_JOBS=${MAX_JOBS} pip wheel --no-build-isolation --no-deps -w /build/dist .
RUN set -eux; WHL="$(find /build -name 'vllm-*.whl' | head -1)"; echo "WHEEL=$WHL"; test -n "$WHL"; \
    pip install --no-cache-dir --no-deps "$WHL"

# This patch changes Python only. Apply it to the installed package after the
# native wheel is built so telemetry edits reuse the expensive CUDA build layer.
COPY patches/mangchi-vllm/0001-live-prefill-and-generation-timings.patch /tmp/vllm-live-timings.patch
RUN cd /opt/venv/lib/python3.12/site-packages \
 && git apply --check /tmp/vllm-live-timings.patch \
 && git apply /tmp/vllm-live-timings.patch

# install vLLM's runtime python deps (xgrammar, etc.) - use_existing_torch already stripped
# torch/flashinfer pins, so this won't downgrade the stack we just built
RUN pip install --no-cache-dir -r requirements/common.txt -r requirements/cuda.txt || true

# flashinfer: the current vLLM pin needs nvidia-cudnn-frontend>=1.25, which has no aarch64 wheel.
# 0.6.8 (anima's) satisfies flashinfer_sampler_supported + the optional flashinfer paths
# without that dep; kernels JIT on first run.
RUN pip install --no-cache-dir --no-deps cuda-tile==1.5.0 \
 && pip install --no-cache-dir --no-deps --extra-index-url https://flashinfer.ai/whl/ flashinfer-python==0.6.8

# transformers v5; compressed-tensors 0.17.x (NVFP4); numba for ngram spec-decode
RUN pip install --no-cache-dir --upgrade transformers
# xgrammar: vLLM 0.29 harmony parser needs >=0.2.1 (openai_tool_call_schema), but no aarch64
# wheel exists past 0.1.32 -> build from source. Built after torch 2.11 so it links correctly.
RUN pip uninstall -y xgrammar || true \
 && pip install --no-cache-dir "pybind11>=2.12" \
 && pip install --no-cache-dir "git+https://github.com/mlc-ai/xgrammar.git@v0.2.6"
RUN pip install --no-cache-dir "compressed-tensors==0.17.1" numba

# vLLM eagerly imports ALL quant backends; guard the ones not packaged for arm64 so
# NVFP4/compressed-tensors still load (humming/inc/fp_quant/torchao/quark/deepseek_v4/mxfp4)
COPY docker/mangchi-vllm/patch_quant.py /tmp/patch_quant.py
RUN python /tmp/patch_quant.py

# Keep Qwen3.8 Flash-Next's 102.4GB BF16 PLE table on SSD. This is a Thor port
# of blazux/qwen3.8-Flash-DGX's mmap loader; it is opt-in via VLLM_PLE_MMAP=1.
COPY docker/mangchi-vllm/vllm_ple_mmap.py /opt/venv/lib/python3.12/site-packages/vllm_ple_mmap.py
COPY docker/mangchi-vllm/patch_ple_mmap.py /tmp/patch_ple_mmap.py
RUN python /tmp/patch_ple_mmap.py

# SM110 supports CUDA clusters in general, but rejects the cooperative top-k launch shape
# used by Qwen3.8's QSA indexer. Route Thor through vLLM's persistent fallback.
COPY docker/mangchi-vllm/patch_thor_qsa.py /tmp/patch_thor_qsa.py
RUN python /tmp/patch_thor_qsa.py

# Optional experimental NVIDIA SM110 W4A4 MoE adapter. Sources are fetched by
# exact revision + SHA256; small native helpers and CuTe kernels compile once
# during model weight preparation, before CUDA graph capture. Build a separate
# canary tag with --build-arg THOR_CUTEDSL_MOE=1 and opt in at runtime with
# VLLM_THOR_CUTEDSL_MOE=1. The default image preserves its existing dependencies.
ARG THOR_CUTEDSL_MOE=0
COPY docker/mangchi-vllm/thor_nvfp4 /opt/venv/lib/python3.12/site-packages/thor_nvfp4
RUN if [ "${THOR_CUTEDSL_MOE}" = "1" ]; then \
      pip install --no-cache-dir --extra-index-url https://pypi.org/simple \
        "nvidia-cutlass-dsl[cu13]==4.7.0" "cupy-cuda13x==13.6.0" \
        "cuda-python==13.3.1" "cuda-bindings==13.3.1" ninja \
      && python -m thor_nvfp4.install --vllm-root /opt/venv/lib/python3.12/site-packages/vllm; \
    elif [ "${THOR_CUTEDSL_MOE}" != "0" ]; then exit 1; fi

# the cloned source tree shadows the installed _C.so -> remove it and serve from /
RUN rm -rf /build
WORKDIR /

# RUNTIME-ONLY: preload the real driver libcuda - the built _C_stable_libtorch.abi3.so has an
# unresolved cuTensorMapEncodeTiled symbol. The driver lib is injected by --runtime nvidia at
# run time, so this MUST be the final instruction (no RUN after it).
ENV LD_PRELOAD=/opt/nvidia/l4t-gpu-libs/openrm/libcuda.so.1.1
