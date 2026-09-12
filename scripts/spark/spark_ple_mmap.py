"""SSD-backed Qwen3.8 Flash-Next PLE embedding for vLLM on DGX Spark.

A port of this repo's Thor loader (docker/mangchi-vllm/vllm_ple_mmap.py, itself
adapted from blazux/qwen3.8-Flash-DGX, Apache-2.0) to the newer vLLM build in
``vllm/vllm-openai:qwen38-flash-next``, where the model moved from
``qwen4_exp`` to ``qwen3_8_flash_next``.

Why this exists: one DGX Spark shares 128 GB between CPU and GPU, and this
checkpoint is 126 GiB. The engine's own ``VLLM_PLE_CPU_OFFLOAD`` moves the
47.7 GiB n-gram table to a host process but still materialises it as a tensor,
so the total does not change. Reading rows from the safetensors shards on NVMe
is what makes a single Spark fit.

Differences from the Thor loader, all forced by the newer build:
  * the placeholder swaps ``VocabParallelEmbedding``; the older build called it
    ``PLEVocabParallelEmbedding``
  * ``__init__`` gained ``max_total_tokens`` and ``max_num_reqs``, so the patch
    forwards arguments untouched instead of restating the signature
  * the lookup lives in ``forward_impl`` and has an ``output_buffer`` fast path
    that indexes ``ngram_embedding.weight`` directly, which a file-backed table
    does not have; the patch routes through the returning path and copies
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

_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F8_E4M3": torch.float8_e4m3fn,
}


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
        "F8_E4M3": 1,
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


def _load_weight_scale(
    embedding: _MmapNgramEmbedding,
    loaded_weight: torch.Tensor,
    device: torch.device,
) -> None:
    if loaded_weight.numel() != 1:
        raise ValueError(
            "FP8 PLE embedding weight_scale must contain exactly one value"
        )
    scale = loaded_weight.to(device=device, dtype=torch.float32)
    if not torch.isfinite(scale).all() or not torch.all(scale > 0):
        raise ValueError("FP8 PLE embedding weight_scale must be positive and finite")
    embedding.register_parameter(
        "weight_scale", nn.Parameter(scale, requires_grad=False)
    )


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


def apply(cls: type) -> None:
    """Patch the n-gram embedding class when explicitly enabled."""
    if not enabled() or getattr(cls, "_ple_mmap_patched", False):
        return
    module = sys.modules[cls.__module__]
    original_init = cls.__init__
    original_load_weights = cls.load_weights
    original_forward_impl = cls.forward_impl

    def patched_init(self, *args, **kwargs):
        # Build the real layer but hand it a placeholder in place of the
        # embedding, so nothing ever allocates the full table.
        real_embedding_class = module.VocabParallelEmbedding
        module.VocabParallelEmbedding = (
            lambda num_embeddings, dim, **kw: _MmapNgramEmbedding(num_embeddings, dim)
        )
        try:
            original_init(self, *args, **kwargs)
        finally:
            module.VocabParallelEmbedding = real_embedding_class

        prefix = kwargs.get("prefix")
        if prefix is None:
            strings = [a for a in args if isinstance(a, str)]
            prefix = strings[0] if strings else ""
        self._ple_mmap_prefix = prefix
        self._ple_mmap_model_path = None
        try:
            from vllm.config import get_current_vllm_config

            self._ple_mmap_model_path = get_current_vllm_config().model_config.model
        except Exception as error:
            logger.warning("PLE mmap: cannot read model path: %s", error)
        params_dtype = kwargs.get("params_dtype")
        if params_dtype is not None:
            self.ngram_embedding._zeros_dtype = params_dtype
        logger.info("PLE mmap: replaced %s with a tiny placeholder", prefix)

    def patched_load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        # Take the table shards and the global scale; everything else still
        # belongs to the engine's own loader.
        remaining = []
        loaded: set[str] = set()
        for name, weight in weights:
            if name.startswith("ngram_embedding.shard_") and name.endswith(".weight"):
                loaded.add("ngram_embedding.weight")
            elif name == "ngram_embedding.weight_scale":
                _load_weight_scale(
                    self.ngram_embedding,
                    weight,
                    self.layer_multipliers.device,
                )
                loaded.add(name)
            else:
                remaining.append((name, weight))
        loaded.update(original_load_weights(self, remaining))
        _setup_table(self)
        return loaded

    def patched_forward_impl(
        self,
        hidden_states,
        input_ids,
        query_start_loc,
        ngram_context,
        output_buffer=None,
    ):
        # The fast path does index_select straight out of a resident weight
        # tensor. There isn't one, so take the returning path and fill the
        # caller's buffer ourselves.
        gathered = original_forward_impl(
            self, hidden_states, input_ids, query_start_loc, ngram_context, None
        )
        if output_buffer is None:
            return gathered
        buffer = output_buffer[: gathered.shape[0], : self.embedding_dim]
        buffer.copy_(gathered)
        return buffer

    cls.__init__ = patched_init
    cls.load_weights = patched_load_weights
    cls.forward_impl = patched_forward_impl
    cls._ple_mmap_patched = True
    logger.info("PLE mmap patch applied to %s.%s", cls.__module__, cls.__name__)
