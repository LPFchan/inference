"""Dependency-free contracts for the first Thor native MoE implementation."""

SOURCE_SHA = "e8b29522938901f6df19ebeedd4b69bc8edbcd97"
VLLM_SHA = "5fd5dd5cf4ac8e9f09b6fae3f3603e9a3cb88aaa"
EXPERTS, HIDDEN, INTERMEDIATE, TOP_K = 512, 2560, 640, 10


def supported(*, sm, quant_method, group_size, hidden, intermediate, experts,
              top_k, activation, dtype, parallel_sizes, has_bias=False,
              lora=False, swiglu_parameters=(None, None, None)):
    return (
        sm == (11, 0) and quant_method == "NVFP4" and group_size == 16
        and (hidden, intermediate, experts, top_k) == (HIDDEN, INTERMEDIATE, EXPERTS, TOP_K)
        and activation == "silu" and dtype == "torch.bfloat16"
        and parallel_sizes == (1, 1, 1, 1, 1)
        and not has_bias and not lora
        and swiglu_parameters == (None, None, None)
    )


def permuted_rows(tokens):
    if not isinstance(tokens, int) or tokens < 0 or tokens > 262144:
        raise ValueError("Thor NVFP4 requires 0 <= token count <= 262144")
    return ((max(1, tokens * TOP_K) + EXPERTS * 127 + 127) // 128) * 128


def fc1_row_order(intermediate=INTERMEDIATE):
    """vLLM [gate, up] -> NVIDIA 64-row [up, gate] chunks, without rounding."""
    if intermediate <= 0 or intermediate % 64:
        raise ValueError("SwiGLU intermediate size must be a positive multiple of 64")
    return [row for start in range(0, intermediate, 64)
            for half in (1, 0)
            for row in range(half * intermediate + start, half * intermediate + start + 64)]


def scale_offset(row, col, rows, cols):
    """Linear [N, K/16] -> NVIDIA [ceil(N/128), ceil(K/64), 32,4,4]."""
    if not (0 <= row < rows and 0 <= col < cols):
        raise ValueError("Scale index outside its logical matrix")
    return (((row // 128 * ((cols + 3) // 4) + col // 4) * 32
             + row % 32) * 4 + row % 128 // 32) * 4 + col % 4
