import importlib.util
import json
import struct
import sys
import types
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

    module._load_weight_scale(
        embedding, "weight_scale", torch.tensor([0.25]), torch.device("cpu")
    )

    assert isinstance(embedding.weight_scale, torch.nn.Parameter)
    assert embedding.weight_scale.dtype is torch.float32
    assert embedding.weight_scale.item() == 0.25


@pytest.mark.parametrize("scale", [torch.tensor([0.0]), torch.tensor([float("nan")])])
def test_fp8_ple_scale_rejects_invalid_values(scale):
    module = _load_module()
    embedding = module._MmapNgramEmbedding(16, 8)

    with pytest.raises(ValueError, match="positive and finite"):
        module._load_weight_scale(embedding, "weight_scale", scale, torch.device("cpu"))


def _write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        data = array.numpy().tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def test_nvfp4_table_gathers_packed_rows_and_block_scales_from_ssd(tmp_path, monkeypatch):
    module = _load_module()
    dim, parts, vocab = 32, 2, 10
    prefix = "model.layers.1.ple.ple_embedding.ngram_embedding."
    generator = torch.Generator().manual_seed(0)
    packed = torch.randint(0, 256, (vocab, dim // 2), dtype=torch.uint8, generator=generator)
    scales = torch.randint(0, 120, (vocab, dim // 16), dtype=torch.uint8, generator=generator)
    tensors = {}
    for index in range(parts):
        rows = slice(index * 5, index * 5 + 5)
        tensors[f"{prefix}shard_{index}.weight"] = ("U8", packed[rows])
        tensors[f"{prefix}shard_{index}.weight_scale"] = ("F8_E4M3", scales[rows])
    _write_safetensors(tmp_path / "model.safetensors", tensors)

    # Reference NVFP4 decode straight from the format: E2M1 nibbles (low nibble
    # is the even element), one E4M3 block scale per 16 values, one global scale.
    def dequantize_rows(rows, row_scales, global_scale, dtype):
        lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
        codes = torch.stack((rows & 0xF, rows >> 4), -1).flatten(-2).long()
        blocks = row_scales.float().repeat_interleave(16, -1)
        return (lut[codes] * blocks * global_scale).to(dtype)

    monkeypatch.setitem(
        sys.modules, "vllm_ple_nvfp4", types.SimpleNamespace(_dequantize_rows=dequantize_rows)
    )
    embedding = module._MmapNgramEmbedding(vocab, dim)
    layer = types.SimpleNamespace(
        ngram_embedding=embedding,
        _ple_mmap_model_path=str(tmp_path),
        _ple_mmap_prefix="model.layers.1.ple",
        head_dim=dim,
        split_ngram_parts=parts,
    )
    # vLLM can deliver the shards before the global scale, in separate calls.
    module._setup_table(layer)
    with pytest.raises(RuntimeError, match="missing its global scale"):
        embedding.gather(torch.tensor([1]))
    module._load_weight_scale(embedding, "weight_scale_2", torch.tensor([2.0]), torch.device("cpu"))

    ids = torch.tensor([[7, 2], [7, 9]])
    got = embedding.gather(ids)

    want = dequantize_rows(packed[ids], scales[ids].view(torch.float8_e4m3fn), 2.0, got.dtype)
    assert got.shape == (2, 2, dim)
    assert torch.equal(got, want)
