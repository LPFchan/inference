// SPDX-License-Identifier: Apache-2.0
// Torch host adapter for NVIDIA's pinned routing and FP4 activation pack kernels.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdint>

namespace {
#include "nvidia/buildLayout.cu.inc"
#include "nvidia/fp4Quantize.cu.inc"
}

void prepare(torch::Tensor x, torch::Tensor ids, torch::Tensor scales,
             torch::Tensor packed, torch::Tensor sf, torch::Tensor mapping,
             torch::Tensor groups, torch::Tensor limits, torch::Tensor tiles) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous());
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 2560 && x.size(0) > 0 && x.size(0) <= 262144);
    c10::cuda::CUDAGuard guard(x.device());
    auto props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 11 && props->minor == 0, "SM110 required");
    for (auto t : {ids, scales, packed, sf, mapping, groups, limits, tiles}) {
        TORCH_CHECK(t.device() == x.device() && t.is_contiguous());
    }
    int m = x.size(0), rows = m * 10;
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.dim() == 2 && ids.size(0) == m && ids.size(1) == 10);
    TORCH_CHECK(scales.scalar_type() == at::kFloat && scales.numel() == 512);
    TORCH_CHECK(packed.scalar_type() == at::kByte && packed.numel() >= int64_t(rows) * 1280);
    TORCH_CHECK(sf.scalar_type() == at::kByte && sf.numel() >= int64_t(rows) * 160);
    int padded = ((rows + 512 * 127 + 127) / 128) * 128;
    TORCH_CHECK(mapping.scalar_type() == at::kInt && mapping.numel() >= padded);
    TORCH_CHECK(groups.scalar_type() == at::kInt && groups.numel() >= padded / 128);
    TORCH_CHECK(limits.scalar_type() == at::kInt && limits.numel() >= padded / 128);
    TORCH_CHECK(tiles.scalar_type() == at::kInt && tiles.numel() == 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto input = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr());
    auto output = reinterpret_cast<uint32_t*>(packed.data_ptr());
    // General setup keeps one implementation for prefill and decode. The two
    // upstream kernels execute on the caller's stream and use caller-owned buffers.
    build_layout_kernel<<<1, 256, 3 * 512 * sizeof(int32_t), stream>>>(
        ids.data_ptr<int32_t>(), mapping.data_ptr<int32_t>(), groups.data_ptr<int32_t>(),
        limits.data_ptr<int32_t>(), tiles.data_ptr<int32_t>(), rows, 512, 128, padded);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int blocks = std::min(rows, props->multiProcessorCount * (2048 / 320));
    quantizeRoutedToFp4LinearSfKernel<__nv_bfloat16><<<blocks, 320, 0, stream>>>(
        m, 10, 2560, input, ids.data_ptr<int32_t>(), scales.data_ptr<float>(), output, sf.data_ptr<uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare", &prepare);
}
