"""SSD-backed Qwen4Exp PLE embedding for vLLM on memory-constrained systems.

Adapted from blazux/qwen3.8-Flash-DGX's Apache-2.0 ``vllm_ple_mmap.py``.
The table stays in safetensors files and random rows are read through mmap.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import mmap
import os
import re
import struct
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn

logger = logging.getLogger("vllm.ple_mmap")

_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16}
_REGISTRY: dict[str, nn.Module] = {}


def enabled() -> bool:
    return os.environ.get("VLLM_PLE_MMAP", "0").lower() in {"1", "true", "yes"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def parse_safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


def _itemsize(dtype: str) -> int:
    return {
        "BF16": 2,
        "F16": 2,
        "F32": 4,
    }[dtype]


class MmapPleTable:
    def __init__(
        self,
        shards: dict[int, tuple[str, int, int]],
        shard_size: int,
        row_bytes: int,
        torch_dtype: torch.dtype,
        workers: int,
        chunk: int,
    ) -> None:
        if not shards:
            raise ValueError("no PLE shards")
        self.shard_size = int(shard_size)
        self.row_bytes = int(row_bytes)
        self.torch_dtype = torch_dtype
        self.chunk = max(1, int(chunk))
        self.paths: list[str | None] = [None] * (max(shards) + 1)
        self.maps: list[np.memmap | None] = [None] * (max(shards) + 1)
        self.rows_total = 0
        advise = os.environ.get("VLLM_PLE_MMAP_MADVISE", "random").lower()
        for index, (path, offset, rows) in shards.items():
            table = np.memmap(
                path,
                dtype=np.uint8,
                mode="r",
                offset=offset,
                shape=(rows, row_bytes),
            )
            if advise in {"random", "1"}:
                raw_map = getattr(table, "_mmap", None)
                if raw_map is not None:
                    try:
                        raw_map.madvise(mmap.MADV_RANDOM)
                    except (AttributeError, OSError):
                        logger.warning("PLE mmap: MADV_RANDOM failed for %s", path)
            self.paths[index] = path
            self.maps[index] = table
            self.rows_total += rows
        self.pool = ThreadPoolExecutor(max_workers=max(1, workers))
        self.fast_rows = _env_int("VLLM_PLE_MMAP_FAST_ROWS", 512)

    def gather(self, ids: np.ndarray) -> np.ndarray:
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.empty((0, self.row_bytes), dtype=np.uint8)
        if ids.min() < 0 or ids.max() >= self.rows_total:
            raise IndexError(
                f"PLE row id range [{ids.min()}, {ids.max()}] exceeds "
                f"{self.rows_total} rows"
            )
        if ids.size <= self.fast_rows:
            shard_ids = ids // self.shard_size
            local_ids = ids - shard_ids * self.shard_size
            output = np.empty((ids.size, self.row_bytes), dtype=np.uint8)
            for shard_id in np.unique(shard_ids):
                mask = shard_ids == shard_id
                shard = self.maps[int(shard_id)]
                if shard is None:
                    raise IndexError(f"PLE shard {shard_id} is missing")
                output[mask] = shard[local_ids[mask]]
            return output

        unique_ids, inverse = np.unique(ids, return_inverse=True)
        shard_ids = unique_ids // self.shard_size
        local_ids = unique_ids - shard_ids * self.shard_size
        output = np.empty((unique_ids.size, self.row_bytes), dtype=np.uint8)
        bounds = np.flatnonzero(np.diff(shard_ids)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [unique_ids.size]))
        tasks: list[tuple[int, int, int]] = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            shard_id = int(shard_ids[start])
            for offset in range(start, end, self.chunk):
                tasks.append((shard_id, offset, min(offset + self.chunk, end)))

        def read_rows(task: tuple[int, int, int]) -> None:
            shard_id, start, end = task
            shard = self.maps[shard_id]
            if shard is None:
                raise IndexError(f"PLE shard {shard_id} is missing")
            output[start:end] = shard[local_ids[start:end]]

        if len(tasks) == 1:
            read_rows(tasks[0])
        else:
            list(self.pool.map(read_rows, tasks))
        return output[inverse]

    def prewarm(self) -> None:
        block_size = 64 << 20
        buffer = bytearray(block_size)
        for path, table in zip(self.paths, self.maps):
            if path is None or table is None:
                continue
            remaining = table.shape[0] * table.shape[1]
            with open(path, "rb", buffering=0) as handle:
                handle.seek(table.offset)
                while remaining:
                    read = handle.readinto(
                        memoryview(buffer)[: min(block_size, remaining)]
                    )
                    if not read:
                        break
                    remaining -= read


class _MmapNgramEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.org_vocab_size = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.table: MmapPleTable | None = None
        self._zeros_dtype = torch.bfloat16

    def _pinned_buffer(self, rows: int, row_bytes: int) -> torch.Tensor | None:
        buffer = getattr(self, "_pinned", None)
        if buffer is None or buffer.shape[0] < rows or buffer.shape[1] != row_bytes:
            try:
                capacity = max(rows + rows // 2, 4096)
                buffer = torch.empty(
                    (capacity, row_bytes), dtype=torch.uint8, pin_memory=True
                )
            except RuntimeError:
                buffer = None
            self._pinned = buffer
        return buffer

    def gather(self, ids: torch.Tensor) -> torch.Tensor:
        if self.table is None:
            return torch.zeros(
                (*ids.shape, self.embedding_dim),
                dtype=self._zeros_dtype,
                device=ids.device,
            )
        ids_cpu = ids.detach().to("cpu", non_blocking=False).numpy().reshape(-1)
        unique_ids, inverse = np.unique(ids_cpu, return_inverse=True)
        rows = self.table.gather(unique_ids)
        count = rows.shape[0]
        staging = (
            self._pinned_buffer(count, self.table.row_bytes)
            if ids.device.type == "cuda"
            else None
        )
        if staging is not None:
            staging[:count].numpy()[:] = rows
            device_rows = staging[:count].to(ids.device, non_blocking=True)
        else:
            device_rows = torch.from_numpy(rows).to(ids.device)
        inverse_tensor = torch.from_numpy(inverse).to(ids.device, non_blocking=True)
        gathered = device_rows.view(self.table.torch_dtype)[inverse_tensor]
        return gathered.reshape(*ids.shape, self.embedding_dim)

    forward = gather


def _find_shards(
    model_path: str, layer_idx: int
) -> tuple[dict[int, tuple[str, int, int]], str | None, int | None]:
    shard_pattern = re.compile(
        rf"layers\.{layer_idx}\.ple\.ple_embedding\.ngram_embedding\."
        r"shard_(\d+)\.weight$"
    )
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as handle:
            weight_map = json.load(handle)["weight_map"]
        files = sorted(
            {
                os.path.join(model_path, filename)
                for name, filename in weight_map.items()
                if shard_pattern.search(name)
            }
        )
    else:
        files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

    shards: dict[int, tuple[str, int, int]] = {}
    dtype: str | None = None
    columns: int | None = None
    for path in files:
        header, data_start = parse_safetensors_header(path)
        for name, metadata in header.items():
            match = shard_pattern.search(name)
            if not match:
                continue
            start, end = metadata["data_offsets"]
            rows, current_columns = metadata["shape"]
            current_dtype = metadata["dtype"]
            if dtype is not None and current_dtype != dtype:
                raise ValueError("PLE shards have mixed dtypes")
            if columns is not None and current_columns != columns:
                raise ValueError("PLE shards have mixed row widths")
            if end - start != rows * current_columns * _itemsize(current_dtype):
                raise ValueError(f"PLE shard {name} has an invalid byte size")
            dtype = current_dtype
            columns = current_columns
            shards[int(match.group(1))] = (path, data_start + start, rows)
    return shards, dtype, columns


def _setup_table(module: nn.Module) -> None:
    embedding = module.ngram_embedding
    if embedding.table is not None:
        return
    model_path = os.environ.get("VLLM_PLE_MMAP_DIR") or module._ple_mmap_model_path
    if not model_path or not os.path.isdir(model_path):
        raise RuntimeError(
            f"PLE mmap path {model_path!r} is not a local directory; use a local "
            "--model path or set VLLM_PLE_MMAP_DIR"
        )
    match = re.search(r"layers\.(\d+)\.", module._ple_mmap_prefix)
    if not match:
        raise RuntimeError(
            f"cannot extract the PLE layer index from {module._ple_mmap_prefix!r}"
        )
    layer_idx = int(match.group(1))
    shards, dtype, columns = _find_shards(model_path, layer_idx)
    if not shards or dtype not in _DTYPES or columns != module.head_dim:
        raise RuntimeError(
            f"invalid PLE mmap table: shards={len(shards)}, dtype={dtype}, "
            f"columns={columns}, expected_columns={module.head_dim}"
        )
    parts = int(module.split_ngram_parts)
    vocab_size = int(embedding.org_vocab_size)
    shard_size = math.ceil(vocab_size / parts)
    expected_indexes = set(range(parts))
    if set(shards) != expected_indexes:
        missing = sorted(expected_indexes - set(shards))
        extra = sorted(set(shards) - expected_indexes)
        raise RuntimeError(f"PLE mmap shard set mismatch: missing={missing}, extra={extra}")
    for index, (_, _, rows) in shards.items():
        expected = max(0, min(shard_size, vocab_size - index * shard_size))
        if rows != expected:
            raise RuntimeError(
                f"PLE shard {index} has {rows} rows; expected {expected}"
            )
    table = MmapPleTable(
        shards,
        shard_size,
        columns * _itemsize(dtype),
        _DTYPES[dtype],
        workers=_env_int("VLLM_PLE_MMAP_WORKERS", 32),
        chunk=_env_int("VLLM_PLE_MMAP_CHUNK", 2048),
    )
    if _env_int("VLLM_PLE_MMAP_PREWARM", 0):
        logger.info(
            "PLE mmap: prewarming %.1f GiB",
            table.rows_total * table.row_bytes / 2**30,
        )
        table.prewarm()
    embedding.table = table
    logger.info(
        "PLE mmap: layer %d uses %d SSD-backed shards, %.1f GiB, dtype %s",
        layer_idx,
        len(shards),
        table.rows_total * table.row_bytes / 2**30,
        dtype,
    )


def _gather_impl(ids: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    output.copy_(_REGISTRY[layer_name].ngram_embedding.gather(ids))


def _gather_fake(ids: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    return


def apply(cls: type) -> None:
    """Patch vLLM's Qwen4ExpNGramEmbedding when explicitly enabled."""
    if not enabled() or getattr(cls, "_ple_mmap_patched", False):
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    if not hasattr(torch.ops.vllm, "qwen4_exp_ple_mmap_gather"):
        direct_register_custom_op(
            op_name="qwen4_exp_ple_mmap_gather",
            op_func=_gather_impl,
            mutates_args=["output"],
            fake_impl=_gather_fake,
        )

    module = sys.modules[cls.__module__]
    original_init = cls.__init__
    original_load_weights = cls.load_weights

    def patched_init(
        self,
        config,
        embedding_dim,
        ple_dense_layer_id,
        max_total_tokens,
        max_num_reqs,
        prefix,
        layer_name,
        quant_config=None,
        params_dtype=None,
    ):
        real_embedding_class = module.PLEVocabParallelEmbedding
        module.PLEVocabParallelEmbedding = (
            lambda num_embeddings, dim, **kwargs: _MmapNgramEmbedding(
                num_embeddings, dim
            )
        )
        try:
            original_init(
                self,
                config,
                embedding_dim,
                ple_dense_layer_id,
                max_total_tokens,
                max_num_reqs,
                prefix,
                layer_name,
                quant_config=None,
                params_dtype=params_dtype,
            )
        finally:
            module.PLEVocabParallelEmbedding = real_embedding_class
        self._ple_mmap_prefix = prefix
        self._ple_mmap_model_path = None
        _REGISTRY[layer_name] = self
        try:
            from vllm.config import get_current_vllm_config

            self._ple_mmap_model_path = get_current_vllm_config().model_config.model
        except Exception as error:
            logger.warning("PLE mmap: cannot read model path: %s", error)
        if params_dtype is not None:
            self.ngram_embedding._zeros_dtype = params_dtype
        logger.info("PLE mmap: replaced %s with a tiny placeholder", prefix)

    def patched_load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        remaining = []
        loaded: set[str] = set()
        for name, weight in weights:
            if name.startswith("ngram_embedding.shard_") and name.endswith(".weight"):
                loaded.add("ngram_embedding.weight")
            else:
                remaining.append((name, weight))
        loaded.update(original_load_weights(self, remaining))
        _setup_table(self)
        return loaded

    def patched_forward(self, input_ids, query_start_loc, ngram_context):
        ids = input_ids.new_empty(
            (input_ids.shape[0], self.ngram_heads), dtype=torch.long
        )
        torch.ops.vllm.qwen4_exp_compute_ple_ngram_ids(
            input_ids, query_start_loc, ngram_context, ids, self.layer_name
        )
        table = self.ngram_embedding.table
        dtype = (
            table.torch_dtype
            if table is not None
            else self.ngram_embedding._zeros_dtype
        )
        output = torch.empty(
            (*ids.shape, self.head_dim), dtype=dtype, device=ids.device
        )
        torch.ops.vllm.qwen4_exp_ple_mmap_gather(ids, output, self.layer_name)
        return output.flatten(-2)

    cls.__init__ = patched_init
    cls.load_weights = patched_load_weights
    cls.forward = patched_forward
    cls._ple_mmap_patched = True
    logger.info("PLE mmap patch applied to %s.%s", cls.__module__, cls.__name__)
