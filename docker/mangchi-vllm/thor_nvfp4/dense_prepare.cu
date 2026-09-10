// SPDX-License-Identifier: Apache-2.0
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
#include "nvidia/fp4Quantize.cu.inc"
}

void quantize(torch::Tensor x, torch::Tensor scale, torch::Tensor packed, torch::Tensor sf) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous());
    TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(0) <= 262144);
    int m = x.size(0), k = x.size(1), padded_m = (m + 127) / 128 * 128;
    TORCH_CHECK(k == 5120 || k == 6144 || k == 17408);
    c10::cuda::CUDAGuard guard(x.device());
    auto props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 11 && props->minor == 0, "SM110 required");
    for (auto t : {scale, packed, sf}) {
        TORCH_CHECK(t.device() == x.device() && t.is_contiguous());
    }
    TORCH_CHECK(scale.scalar_type() == at::kFloat && scale.numel() == 1);
    TORCH_CHECK(packed.scalar_type() == at::kByte && packed.numel() == int64_t(m) * k / 2);
    TORCH_CHECK(sf.scalar_type() == at::kByte && sf.numel() == int64_t(padded_m) * k / 16);
    int blocks = std::min(padded_m, props->multiProcessorCount * 8);
    quantizeToFp4Kernel<__nv_bfloat16><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        m, k, reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()), scale.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(packed.data_ptr()), sf.data_ptr<uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("quantize", &quantize); }
