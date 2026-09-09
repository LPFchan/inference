import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "docker/mangchi-vllm/vllm_ple_mmap.py"
    spec = importlib.util.spec_from_file_location("vllm_ple_mmap_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fp8_e4m3_safetensors_dtype_is_supported():
    module = _load_module()

    assert module._itemsize("F8_E4M3") == 1
    assert module._DTYPES["F8_E4M3"] is torch.float8_e4m3fn


def test_fp8_ple_scale_is_registered_for_runtime_dequantization():
    module = _load_module()
    embedding = module._MmapNgramEmbedding(16, 8)

    module._load_weight_scale(embedding, torch.tensor([0.25]), torch.device("cpu"))

    assert isinstance(embedding.weight_scale, torch.nn.Parameter)
    assert embedding.weight_scale.dtype is torch.float32
    assert embedding.weight_scale.item() == 0.25


@pytest.mark.parametrize("scale", [torch.tensor([0.0]), torch.tensor([float("nan")])])
def test_fp8_ple_scale_rejects_invalid_values(scale):
    module = _load_module()
    embedding = module._MmapNgramEmbedding(16, 8)

    with pytest.raises(ValueError, match="positive and finite"):
        module._load_weight_scale(embedding, scale, torch.device("cpu"))
