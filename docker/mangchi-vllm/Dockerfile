# mangchi-vllm:thor-v0.29-ple-mmap - vLLM 0.29.0 on Jetson AGX Thor (sm_110a)
# Adapted from RobotFlow-Labs/anima-vllm-thor (Apache-2.0), bumped v0.23.0 -> v0.29.0.
# v0.29.0 is the first vLLM ref that registers BOTH qwen3_5 (our 27B) and qwen4_exp
# (Flash-Next). Built on NVIDIA's Thor base (CUDA 13.0, sm_110a kernels).
# Build: ~30-50 min, needs most of the 128GB unified memory free. Don't build while serving.
#   DOCKER_BUILDKIT=1 docker build -t mangchi-vllm:thor-v0.29-ple-mmap -f Dockerfile .
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

# build vLLM v0.29.0 from source against the installed torch (reuse it, strip torch pins)
WORKDIR /build
RUN git clone --depth 1 --branch v0.29.0 https://github.com/vllm-project/vllm.git
WORKDIR /build/vllm
RUN python use_existing_torch.py || true
RUN pip install --no-cache-dir "setuptools-scm>=8"
RUN MAX_JOBS=${MAX_JOBS} pip wheel --no-build-isolation --no-deps -w /build/dist .
RUN set -eux; WHL="$(find /build -name 'vllm-*.whl' | head -1)"; echo "WHEEL=$WHL"; test -n "$WHL"; \
    pip install --no-cache-dir --no-deps "$WHL"

# install vLLM's runtime python deps (xgrammar, etc.) - use_existing_torch already stripped
# torch/flashinfer pins, so this won't downgrade the stack we just built
RUN pip install --no-cache-dir -r requirements/common.txt -r requirements/cuda.txt || true

# flashinfer: 0.6.18 (v0.29 pin) needs nvidia-cudnn-frontend>=1.25 which has no aarch64 wheel.
# 0.6.8 (anima's) satisfies flashinfer_sampler_supported + the optional flashinfer paths
# without that dep; kernels JIT on first run.
RUN pip install --no-cache-dir --extra-index-url https://flashinfer.ai/whl/ flashinfer-python==0.6.8 || true

# transformers v5 (v0.29 dropped v4 support); compressed-tensors 0.17.x (NVFP4); numba for ngram spec-decode
RUN pip install --no-cache-dir --upgrade transformers
# xgrammar: vLLM 0.29 harmony parser needs >=0.2.1 (openai_tool_call_schema), but no aarch64
# wheel exists past 0.1.32 -> build from source. Built after torch 2.13 so it links correctly.
RUN pip uninstall -y xgrammar || true \
 && pip install --no-cache-dir "pybind11>=2.12" \
 && pip install --no-cache-dir "git+https://github.com/mlc-ai/xgrammar.git@v0.2.6"
RUN pip install --no-cache-dir "compressed-tensors==0.17.1" numba

# vLLM eagerly imports ALL quant backends; guard the ones not packaged for arm64 so
# NVFP4/compressed-tensors still load (humming/inc/fp_quant/torchao/quark/deepseek_v4/mxfp4)
COPY patch_quant.py /tmp/patch_quant.py
RUN python /tmp/patch_quant.py

# Keep Qwen3.8 Flash-Next's 102.4GB BF16 PLE table on SSD. This is a Thor port
# of blazux/qwen3.8-Flash-DGX's mmap loader; it is opt-in via VLLM_PLE_MMAP=1.
COPY vllm_ple_mmap.py /opt/venv/lib/python3.12/site-packages/vllm_ple_mmap.py
COPY patch_ple_mmap.py /tmp/patch_ple_mmap.py
RUN python /tmp/patch_ple_mmap.py

# SM110 supports CUDA clusters in general, but rejects the cooperative top-k launch shape
# used by Qwen3.8's QSA indexer. Route Thor through vLLM's existing persistent fallback.
COPY patch_thor_qsa.py /tmp/patch_thor_qsa.py
RUN python /tmp/patch_thor_qsa.py


# the cloned source tree shadows the installed _C.so -> remove it and serve from /
RUN rm -rf /build
WORKDIR /

# RUNTIME-ONLY: preload the real driver libcuda - the built _C_stable_libtorch.abi3.so has an
# unresolved cuTensorMapEncodeTiled symbol. The driver lib is injected by --runtime nvidia at
# run time, so this MUST be the final instruction (no RUN after it).
ENV LD_PRELOAD=/opt/nvidia/l4t-gpu-libs/openrm/libcuda.so.1.1
