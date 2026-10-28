#!/usr/bin/env python3
"""Export the public Nemotron Transcribe release to bounded-memory MLX INT4."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

SOURCE_ID = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
SOURCE_REVISION = "a188fff16c4f129a046788c2ad69e6198079a102"
SOURCE_SHA256 = "03142dde5176c1d4df21ee829642745a1d670ea4d79e1a4684c64d53195acff2"
IMPLEMENTATION = "c1e429151c56ce4057c62709f958df998e301d3f"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source / "model.safetensors"
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("Source weights do not match the pinned public release")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Export directory must be empty")

    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_audio.stt.models.nemotron_transcribe import Model, ModelConfig

    # Instantiate lazy shapes only. Never materialize the 67 GB BF16 model.
    config = json.loads((args.source / "config.json").read_text())
    model = Model(ModelConfig.from_dict(config))
    expected = {key: value.shape for key, value in tree_flatten(model.parameters())}
    quantizable = {
        name + ".weight" for name, module in model.named_modules()
        if name.startswith("llm.") and hasattr(module, "to_quantized")
        and module.weight.shape[-1] % 64 == 0
        and not name.endswith(".gate")
    }
    del model
    mx.set_cache_limit(128 * 1024**2)
    weights = mx.load(str(source))
    omitted = [key for key in weights if key.startswith(("ctc_timestamp.", "llm.mtp."))
               or key.endswith("._extra_state")]
    for key in omitted:
        del weights[key]
    weights = Model.sanitize(weights)
    actual = {key: value.shape for key, value in weights.items()}
    if actual != expected:
        missing, extra = set(expected) - set(actual), set(actual) - set(expected)
        wrong = {key for key in expected.keys() & actual.keys() if expected[key] != actual[key]}
        raise ValueError(f"Release/port shape mismatch: missing={missing}, extra={extra}, shapes={wrong}")

    # Evaluate and serialize one tensor at a time; keep at most a 512 MiB shard.
    args.output.mkdir(parents=True, exist_ok=True)
    shards, shard, size, total = [], {}, 0, 0
    weight_map = {}

    def flush():
        nonlocal shard, size
        if not shard:
            return
        name = f"model-{len(shards) + 1:05d}.safetensors"
        mx.save_safetensors(str(args.output / name), shard, metadata={"format": "mlx"})
        weight_map.update({key: name for key in shard})
        shards.append(name)
        shard, size = {}, 0
        mx.clear_cache()

    for index, key in enumerate(list(weights)):
        value = weights.pop(key)
        if key in quantizable:
            q, scales, biases = mx.quantize(value, group_size=64, bits=4, mode="affine")
            mx.eval(q, scales, biases)
            converted = {key: q, key[:-7] + ".scales": scales, key[:-7] + ".biases": biases}
        else:
            mx.eval(value)
            converted = {key: value}
        del value
        for name, array in converted.items():
            if size + array.nbytes > 512 * 1024**2:
                flush()
            shard[name] = array
            size += array.nbytes
            total += array.nbytes
        del converted
        if index % 50 == 0:
            print(f"Converted {index + 1}/{len(actual)} tensors; {total / 2**30:.2f} GiB", flush=True)
        mx.clear_cache()
    flush()
    (args.output / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    config["model_type"] = "nemotron_transcribe"
    config["quantization"] = config["quantization_config"] = {"bits": 4, "group_size": 64, "mode": "affine"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    for name in ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "LICENSE", "THIRD_PARTY_NOTICES.md"]:
        shutil.copy2(args.source / name, args.output / name)
    shutil.copytree(args.source / "licenses", args.output / "licenses", dirs_exist_ok=True)
    evidence = {
        "source_id": SOURCE_ID, "source_revision": SOURCE_REVISION, "source_sha256": SOURCE_SHA256,
        "implementation_commit": IMPLEMENTATION, "quantization": config["quantization"],
        "perception_quantized": False, "router_quantized": False,
        "omitted": omitted, "parameter_bytes": total,
        "files": {path.name: sha256(path) for path in args.output.iterdir() if path.is_file()},
    }
    (args.output / "conversion.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps({"parameter_bytes": total, "quantized_modules": len(quantizable), "peak_memory_bytes": mx.get_peak_memory()}), flush=True)


if __name__ == "__main__":
    main()
