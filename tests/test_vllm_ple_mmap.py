import importlib.util
from pathlib import Path

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
