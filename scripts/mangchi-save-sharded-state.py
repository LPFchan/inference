#!/usr/bin/env python3
"""Materialize vLLM's post-load target weights as a fast sharded checkpoint.

Run this inside the Mangchi vLLM image with the model directory mounted
read-write. The source checkpoint remains unchanged; the destination receives
the prepared tensors plus the tokenizer and processor metadata needed to serve.
The export intentionally uses the production 393K/MTP4 geometry.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from vllm import LLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--max-shard-size-gib", type=int, default=4)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    destination = Path(args.destination).resolve()
    if not source.is_dir():
        parser.error(f"source model directory does not exist: {source}")
    if destination.exists() and any(destination.iterdir()):
        parser.error(f"destination must be absent or empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=str(source),
        max_model_len=393216,
        max_num_seqs=4,
        max_num_batched_tokens=8192,
        enforce_eager=True,
        enable_flashinfer_autotune=False,
        enable_prefix_caching=True,
        moe_backend="auto",
        skip_mm_profiling=True,
        kv_cache_dtype="fp8",
        kv_cache_memory_bytes=14485411127,
        speculative_config={
            "method": "mtp",
            "model": str(source),
            "num_speculative_tokens": 4,
            "max_model_len": 393216,
        },
        hf_overrides={
            "rope_scaling": {
                "type": "yarn",
                "factor": 1.5,
                "original_max_position_embeddings": 262144,
            }
        },
    )
    llm.llm_engine.engine_core.save_sharded_state(
        str(destination), max_size=args.max_shard_size_gib << 30
    )

    metadata = (
        "config.json",
        "hf_quant_config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    )
    for name in metadata:
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, destination / name)

    shards = sorted(destination.glob("model-rank-0-part-*.safetensors"))
    if not shards:
        raise RuntimeError(f"vLLM did not write sharded state into {destination}")
    total = sum(path.stat().st_size for path in shards)
    print(f"wrote {len(shards)} shards ({total / 2**30:.2f} GiB) to {destination}")


if __name__ == "__main__":
    main()
