# syntax=docker/dockerfile:1.7

# maya-ik2: Thinking Machines' Inkling-2-Mini dedicated CUDA engine (sm_86, RTX 3090)
# Upstream: https://github.com/alesha-pro/maya-inkling-mini
# Built on CUDA 12.8 + Ubuntu 24.04 (cmake 3.28, Python 3.12).

ARG MAYA_REPO_URL=https://github.com/alesha-pro/maya-inkling-mini.git
ARG MAYA_SHA=e819852f822479af783638a467bfdfe2ed6882ec
ARG LLAMA_REPO_URL=https://github.com/ggml-org/llama.cpp.git
# llama.cpp tag b12590 (pinned by maya-inkling-mini for maya-vision mtmd)
ARG LLAMA_SHA=34ff2e047ce520b1d83bd274bf3ff0bfbc72a69b
ARG CUDA_ARCHITECTURES=86

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG MAYA_REPO_URL
ARG MAYA_SHA
ARG LLAMA_REPO_URL
ARG LLAMA_SHA
ARG CUDA_ARCHITECTURES

RUN git init . \
    && git remote add origin "${MAYA_REPO_URL}" \
    && git fetch --depth 1 origin "${MAYA_SHA}" \
    && git checkout --detach "${MAYA_SHA}" \
    && test "$(git rev-parse HEAD)" = "${MAYA_SHA}"

COPY patches /tmp/maya-patches
RUN for p in /tmp/maya-patches/*.patch; do \
        echo "Applying $p..." && git apply "$p"; \
    done && rm -rf /tmp/maya-patches

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_ENABLE_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DSTRATA_BUILD_TESTS=OFF \
    && cmake --build build -j"$(nproc)" --target maya-ik2

RUN git init third_party/llama.cpp-b12590 \
    && git -C third_party/llama.cpp-b12590 remote add origin "${LLAMA_REPO_URL}" \
    && git -C third_party/llama.cpp-b12590 fetch --depth 1 origin "${LLAMA_SHA}" \
    && git -C third_party/llama.cpp-b12590 checkout --detach "${LLAMA_SHA}" \
    && test "$(git -C third_party/llama.cpp-b12590 rev-parse HEAD)" = "${LLAMA_SHA}" \
    && cmake -S tools/vision -B build-vision -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_VISION_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DLLAMA_DIR=/app/third_party/llama.cpp-b12590 \
    && cmake --build build-vision -j"$(nproc)" --target maya-vision

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir \
        numpy \
        jinja2 \
        regex \
        pyyaml \
        tqdm \
        requests \
        pillow \
        psutil \
        "jsonschema>=4.18,<5"

WORKDIR /app

COPY --from=build /app/build/maya-ik2 /app/build/maya-ik2
COPY --from=build /app/build-vision/bin/maya-vision /app/build-vision/bin/maya-vision
COPY --from=build /app/serve /app/serve
COPY --from=build /app/tools /app/tools

# Verify runtime shared libraries resolve cleanly (exclude libcuda.so.1 injected by nvidia runtime).
RUN ldd /app/build/maya-ik2 /app/build-vision/bin/maya-vision | tee /dev/stderr \
    | grep "not found" | grep -v "libcuda.so" && exit 1 || true

COPY ik2-mini.json /app/ik2-mini.json
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 18000

ENTRYPOINT ["/app/entrypoint.sh"]
