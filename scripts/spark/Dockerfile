# vLLM for DGX Spark (GB10, sm_121) at the exact commit Mangchi runs, so a
# Thor-vs-Spark comparison differs only by silicon.
#
#   docker build -t spark-vllm:fn -f scripts/spark/Dockerfile .   # context: repo root
#
# Base is vLLM's day-0 Flash-Next image, kept only for its working aarch64
# dependency set (torch 2.13+cu130, flashinfer, xgrammar). Its vLLM is removed
# and rebuilt from source, which is also what renames the model module from
# qwen3_8_flash_next to qwen4_exp -- and that is the point: at this commit the
# Thor patches in docker/mangchi-vllm/ apply unchanged, and QSA accepts an FP8
# E4M3 main KV cache (upstream PR #55557, dequantised inside the Triton kernel).
# Pinned by digest, not tag: this is a one-off day-0 tag for a single model
# release, and a tag can be repointed or removed. The digest is the
# multi-arch manifest list; docker still selects the arm64 member.
FROM vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

ENV TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    CUDA_HOME=/usr/local/cuda \
    VLLM_TARGET_DEVICE=cuda \
    CMAKE_BUILD_TYPE=Release

# The runtime image ships nvcc, gcc and ninja but not these two.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "cmake>=3.26"

# --no-build-isolation below means every build requirement must already be here.
# This commit builds a Rust extension, hence the toolchain as well as
# setuptools-rust; the same cargo caches appear in the Thor Dockerfile.
RUN pip install --no-cache-dir "setuptools>=77" "setuptools-scm>=8" \
    "setuptools-rust>=1.10" packaging wheel ninja
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/root/.cargo/bin:$PATH

# Leave the dependency set alone; only vLLM itself is replaced.
RUN pip uninstall -y vllm || true

# The runtime image ships versioned CUDA libraries but not all of the bare .so
# development symlinks CMake's FindCUDAToolkit looks for; libnvrtc is the one
# vLLM needs (spinloop, cumem_allocator and fs_io_C link against it).
RUN set -eux; \
    d=$(dirname "$(find /usr/local/cuda*/targets -name 'libnvrtc.so.13' | head -1)"); \
    for so in "$d"/lib*.so.13; do \
      base="${so%.so.13}.so"; [ -e "$base" ] || ln -s "$so" "$base"; \
    done; \
    test -e "$d/libnvrtc.so"; \
    # torch's CUDAContextLight.h includes cusparse.h, which this image ships only
    # in the pip CUDA tree. Link those headers in one by one: putting that whole
    # include directory on the search path instead drags in a second copy of CCCL
    # and the crt headers, and the two trees miscompile against each other
    # ("__cudaLaunch passed 2 arguments", cuda::std::max overload errors).
    src=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include; \
    dst=$(dirname "$(find /usr/local/cuda*/targets -name 'cuda_runtime.h' | head -1)"); \
    for h in "$src"/cusparse*.h; do \
      [ -e "$dst/$(basename "$h")" ] || ln -s "$h" "$dst/$(basename "$h")"; \
    done; \
    test -e "$dst/cusparse.h"

ARG VLLM_REF=refs/pull/55557/head
ARG VLLM_SHA=5fd5dd5cf4ac8e9f09b6fae3f3603e9a3cb88aaa
WORKDIR /build
# Fetch the SHA directly. The PR ref is only a fallback, because upstream has
# already moved it past the commit Mangchi pinned -- checking out FETCH_HEAD
# would silently build a different engine than Thor runs.
RUN git init vllm \
 && git -C vllm remote add origin https://github.com/vllm-project/vllm.git \
 && (git -C vllm fetch --depth 1 origin "${VLLM_SHA}" \
     || git -C vllm fetch --depth 200 origin "${VLLM_REF}") \
 && git -C vllm checkout --detach "${VLLM_SHA}" \
 && test "$(git -C vllm rev-parse HEAD)" = "${VLLM_SHA}"

# Reuse Mangchi's target-skipping patch. The option is named for Thor but it is
# a plain switch over vLLM's external CUDA projects, and every one it drops is
# either unsupported here or unwanted: FlashMLA already reports "unsupported
# CUDA architecture", fmha_sm100 targets a different family, and DeepGEMM is the
# kernel others disable outright on GB10 (vllm#54125). FA3 is Hopper; this model
# runs FA2.
COPY patches/mangchi-vllm/0003-thor-minimal-vllm-targets.patch /tmp/minimal-targets.patch
RUN git -C /build/vllm apply --check /tmp/minimal-targets.patch \
 && git -C /build/vllm apply /tmp/minimal-targets.patch

WORKDIR /build/vllm
# Build against the torch already in the image rather than letting the pin drag
# in a different one; the base's torch is the aarch64 build that works here.
RUN python3 use_existing_torch.py || true
ARG MAX_JOBS=16
RUN --mount=type=cache,id=spark-vllm-build,target=/build/vllm/build \
    --mount=type=cache,id=spark-vllm-deps,target=/build/vllm/.deps \
    --mount=type=cache,id=spark-vllm-cargo-registry,target=/root/.cargo/registry \
    find build -type f -name CMakeCache.txt -delete; \
    VLLM_THOR_MINIMAL_BUILD=1 \
    CMAKE_ARGS="-DVLLM_THOR_MINIMAL_BUILD=ON" \
    MAX_JOBS=${MAX_JOBS} pip wheel --no-build-isolation --no-deps -w /build/dist .
RUN set -eux; WHL="$(find /build/dist -name 'vllm-*.whl' | head -1)"; test -n "$WHL"; \
    pip install --no-cache-dir --no-deps "$WHL"

# Drop the build tree before patching. While /build/vllm is the working
# directory, "import vllm" resolves to the source checkout, so the patches would
# land there and then be deleted with it -- and the checkout also shadows the
# installed _C.so at runtime.
RUN rm -rf /build
WORKDIR /

# The Mangchi patches, unmodified. They locate site-packages through sysconfig,
# which points at the venv on Thor's base and at the wrong tree here, so the
# path is rewritten to wherever vllm actually landed.
COPY docker/mangchi-vllm/vllm_ple_mmap.py /tmp/vllm_ple_mmap.py
COPY docker/mangchi-vllm/patch_ple_mmap.py docker/mangchi-vllm/patch_draft_vocab.py \
     docker/mangchi-vllm/patch_mamba_block_size.py /tmp/
RUN python3 - <<'PY'
import os, pathlib, vllm
root = pathlib.Path(os.path.dirname(os.path.dirname(vllm.__file__)))
(root / "vllm_ple_mmap.py").write_bytes(pathlib.Path("/tmp/vllm_ple_mmap.py").read_bytes())
for name in ("patch_ple_mmap.py", "patch_draft_vocab.py", "patch_mamba_block_size.py"):
    p = pathlib.Path("/tmp") / name
    p.write_text(p.read_text().replace(
        'Path(sysconfig.get_paths()["purelib"])', f'Path("{root}")'))
print("site-packages:", root)
PY
RUN python3 /tmp/patch_ple_mmap.py \
 && python3 /tmp/patch_draft_vocab.py \
 && python3 /tmp/patch_mamba_block_size.py
