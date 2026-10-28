#!/usr/bin/env python3
"""Qwen3-ASR decoder LoRA SFT and serialized, audio-conditioned GRPO on MLX."""
import argparse
import fcntl
import json
import math
import platform
import random
import socket
import time
from pathlib import Path

from common import (LANGUAGES, advantages, atomic_json, load_manifests, reward,
                    recover_uncommitted_updates, sha256, training_targets)

MODEL_ID = "mlx-community/Qwen3-ASR-1.7B-8bit"
MODEL_REVISION = "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"
SCHEDULE = ("ko", "ja", "zh") * 3 + ("en",)


def grpo_loss(log_probs, old_probs, reference_probs, advantage, beta):
    import mlx.core as mx
    ratio = mx.exp(log_probs - old_probs)
    policy = mx.minimum(ratio * advantage, mx.clip(ratio, 0.8, 1.2) * advantage)
    diff = reference_probs - log_probs
    return (-policy + beta * (mx.exp(diff) - diff - 1)).mean()


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("sft", "grpo", "evaluate"))
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--protected-test", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--init-adapter", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--steps", type=int, default=200, help="absolute optimizer-step target")
    p.add_argument("--accumulate", type=int, default=4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--scale", type=float, default=1.0, help="LoRA multiplier (alpha/rank)")
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=0.02)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--eval-clips", type=int, default=4, help="per language on development only")
    p.add_argument("--memory-gb", type=float, default=18.0)
    p.add_argument("--seed", type=int, default=20261029)
    a = p.parse_args()
    if min(a.steps, a.accumulate, a.rank, a.save_every, a.eval_clips, a.max_tokens) < 1:
        p.error("counts must be positive")
    if a.group_size < 2 or a.temperature <= 0 or a.beta < 0 or a.memory_gb <= 0:
        p.error("invalid GRPO/memory settings")
    if a.stage == "grpo" and not a.init_adapter:
        p.error("GRPO requires an SFT checkpoint via --init-adapter")
    a.learning_rate = a.learning_rate or (1e-4 if a.stage == "sft" else 1e-6)
    if a.learning_rate <= 0:
        p.error("positive learning rate required")
    return a


def main():
    a = arguments()
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Training and the 20 GB production model cannot share this Mac.
        try:
            connection = socket.create_connection(("127.0.0.1", 8791), timeout=0.3)
        except OSError:
            pass
        else:
            connection.close()
            raise RuntimeError("STT backend is active; use with-training-slot.sh to pause and restore it")
        run(a)


def run(a):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    from mlx_audio.stt.utils import load_model
    from mlx_audio.lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.lora import LoRALinear

    train, dev, data_meta = load_manifests(a.data, a.protected_test)
    pools = {lang: [r for r in train if r["language"] == lang] for lang in LANGUAGES}
    if any(not pool for pool in pools.values()):
        raise ValueError("all four training languages required")
    model_files = {str(p.relative_to(a.model)): sha256(p) for p in sorted(a.model.rglob("*"))
                   if p.is_file() and ".cache" not in p.parts and not p.name.startswith(".")}
    if not any(name.endswith(".safetensors") for name in model_files):
        raise ValueError("missing model weights")
    source = json.loads((a.model / "training-source.json").read_text())
    if source != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("unexpected base model source")
    init = a.init_adapter.resolve() if a.init_adapter else None
    settings = {key: getattr(a, key) for key in (
        "stage", "accumulate", "rank", "scale", "learning_rate", "group_size",
        "temperature", "beta", "max_tokens", "seed", "memory_gb")}
    contract = {"settings": settings, "model_files": model_files,
                "data": {k: data_meta[k] for k in (
                    "train_sha256", "dev_sha256", "protected_manifest_sha256")},
                "init_adapter_sha256": sha256(init / "adapters.safetensors") if init else None,
                "code": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
                "scorer_sha256": sha256(Path(__file__).resolve().parents[2] / "scripts/stt-score.py")}
    run_file = a.output / "run.json"
    if run_file.exists():
        if not a.resume or json.loads(run_file.read_text())["contract"] != contract:
            raise ValueError("existing run requires --resume and identical inputs/settings/code")
    elif a.resume:
        raise ValueError("no run to resume")
    else:
        atomic_json(run_file, {"contract": contract, "platform": platform.platform(),
                              "started": time.time(), "status": "running"})

    mx.set_memory_limit(int(a.memory_gb * (1 << 30)))
    mx.set_cache_limit(128 << 20)
    mx.random.seed(a.seed)
    outer = load_model(str(a.model))
    model = getattr(outer, "_model", outer)
    model.freeze()
    modules = []
    for name, layer in model.named_modules():
        if name.startswith("model.layers.") and name.rsplit(".", 1)[-1] in {
                "q_proj", "k_proj", "v_proj", "o_proj"}:
            modules.append((name, LoRALinear.from_base(layer, r=a.rank, scale=a.scale)))
    if len(modules) != 112:
        raise ValueError(f"expected 112 decoder projections, got {len(modules)}")
    model.update_modules(tree_unflatten(modules))
    model.freeze()
    for _, layer in modules:
        layer.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
    model.eval()  # dropout-free likelihoods for both training and rollouts
    adapter_names = {name for name, _ in tree_flatten(model.trainable_parameters())}
    if not adapter_names or any(not name.endswith(("lora_a", "lora_b")) for name in adapter_names):
        raise ValueError("unexpected trainable base parameters")
    mx.eval(model.parameters())
    initial = {name: mx.array(value) for name, value in tree_flatten(model.trainable_parameters())}

    def load_adapter(directory):
        info = json.loads((directory / "checkpoint.json").read_text())
        if info["model_files"] != model_files or info["rank"] != a.rank or info["scale"] != a.scale:
            raise ValueError("checkpoint/base or adapter geometry mismatch")
        weights = mx.load(str(directory / "adapters.safetensors"))
        if set(weights) != adapter_names or sha256(directory / "adapters.safetensors") != info["adapters_sha256"]:
            raise ValueError("invalid adapter checkpoint")
        model.load_weights(list(weights.items()), strict=False)
        return info

    if init:
        info = load_adapter(init)
        if a.stage == "grpo" and (info["stage"] != "sft" or info["step"] < 1):
            raise ValueError("RL reference must be a trained SFT checkpoint")
    # Fixed SFT reference uses only an adapter copy, sharing frozen base weights.
    reference = {name: mx.array(value) for name, value in tree_flatten(model.trainable_parameters())}
    optimizer = optim.AdamW(learning_rate=a.learning_rate, weight_decay=0.0)
    start = 0
    if a.resume:
        latest_path = a.output / "latest.json"
        if latest_path.exists():
            latest = json.loads(latest_path.read_text())
            checkpoint = a.output / latest["directory"]
            info = load_adapter(checkpoint)
            if sha256(checkpoint / "optimizer.safetensors") != info["optimizer_sha256"]:
                raise ValueError("changed optimizer state")
            optimizer.state = tree_unflatten(list(mx.load(str(checkpoint / "optimizer.safetensors")).items()))
            start = info["step"]
        recover_uncommitted_updates(a.output, start)
    eos = model._tokenizer.eos_token_id
    if eos not in model._eos_token_ids():
        raise ValueError("tokenizer EOS disagrees with runtime")

    def context(row):
        features, mask, count = model._preprocess_audio(str(a.data / row["audio"]))
        audio = mx.stop_gradient(model.get_audio_features(features, mask))
        prompt = model._build_prompt(count, language=LANGUAGES[row["language"]])
        prompt_ids = prompt[0].tolist()
        embeds = mx.stop_gradient(model._build_inputs_embeds(prompt, audio))
        mx.eval(embeds)
        return prompt_ids, embeds

    def log_probs(m, prompt_ids, prompt_embeds, completion, temperature=1.0):
        # Project transcript positions only, avoiding prefix x 152k-vocab logits.
        inputs, targets, _ = training_targets(prompt_ids, completion)
        suffix = m.model.embed_tokens(mx.array([completion[:-1]], dtype=mx.int32))
        embeds = mx.concatenate([prompt_embeds, suffix], axis=1)
        hidden = m.model(inputs_embeds=embeds)[:, len(prompt_ids) - 1:]
        logits = m.lm_head(hidden) if m.lm_head is not None else m.model.embed_tokens.as_linear(hidden)
        logits = logits.astype(mx.float32) / temperature
        tokens = mx.array([targets[len(prompt_ids) - 1:]], dtype=mx.int32)
        return -nn.losses.cross_entropy(logits, tokens, reduction="none")[0]

    def sample(row, ctx, temperature):
        ids, embeds = ctx
        completion = []
        for token, _ in generate_step(mx.array(ids, dtype=mx.int32), model,
                input_embeddings=embeds[0], max_tokens=a.max_tokens,
                sampler=make_sampler(temp=temperature), prefill_step_size=256):
            token = int(token)
            completion.append(token)
            if token in model._eos_token_ids():
                break
        ended = bool(completion and completion[-1] in model._eos_token_ids())
        text = model._tokenizer.decode(completion[:-1] if ended else completion,
                                       skip_special_tokens=True)
        return completion, text, ended

    def evaluate(tag):
        from collections import defaultdict
        results = []
        counts = defaultdict(int)
        started = time.perf_counter()
        for row in dev:
            language = row["language"]
            if counts[language] >= a.eval_clips:
                continue
            ctx = context(row)
            tokens = model._tokenizer.encode(row["text"], add_special_tokens=False) + [eos]
            loss = float(-log_probs(model, *ctx, tokens).mean().item())
            completion, text, ended = sample(row, ctx, 0.0)
            results.append({"id": row["id"], "language": language, "reference": row["text"],
                            "hypothesis": text, "loss": loss, "ended": ended,
                            "reward": reward(row["text"], text, language)})
            counts[language] += 1
            mx.clear_cache()
        atomic_json(a.output / f"dev-{tag}.json", {"rows": results,
                    "seconds": time.perf_counter() - started,
                    "peak_memory_bytes": mx.get_peak_memory()})

    if start == 0 and not (a.output / "dev-initial.json").exists():
        evaluate("initial")
    if a.stage == "evaluate":
        return

    def checkpoint(step):
        directory = a.output / f"step-{step:06d}"
        directory.mkdir()  # never overwrite a checkpoint
        params = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(directory / "adapters.safetensors"), params)
        mx.save_safetensors(str(directory / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
        change = sum(float(mx.sum(mx.abs(value - initial[name])).item()) for name, value in params.items())
        atomic_json(directory / "checkpoint.json", {
            "stage": a.stage, "step": step, "rank": a.rank, "scale": a.scale,
            "model_files": model_files, "contract": contract,
            "adapters_sha256": sha256(directory / "adapters.safetensors"),
            "optimizer_sha256": sha256(directory / "optimizer.safetensors"),
            "adapter_l1_change_from_random_init": change,
            "trainable_parameters": sum(v.size for v in params.values()),
            "peak_memory_bytes": mx.get_peak_memory()})
        atomic_json(a.output / "latest.json", {"directory": directory.name, "step": step})

    for step in range(start + 1, a.steps + 1):
        began = time.perf_counter()
        mx.random.seed(a.seed + step)
        grads, losses, diagnostics = None, [], {}
        if a.stage == "sft":
            for micro in range(a.accumulate):
                index = (step - 1) * a.accumulate + micro
                lang = SCHEDULE[index % len(SCHEDULE)]
                row = random.Random(a.seed + index).choice(pools[lang])
                ctx = context(row)
                completion = model._tokenizer.encode(row["text"], add_special_tokens=False) + [eos]
                loss, grad = nn.value_and_grad(model, lambda m: -log_probs(m, *ctx, completion).mean())(model)
                mx.eval(loss, grad)
                grads = grad if grads is None else tree_map(lambda x, y: x + y, grads, grad)
                losses.append(float(loss.item()))
            grads = tree_map(lambda g: g / a.accumulate, grads)
        else:
            lang = SCHEDULE[(step - 1) % len(SCHEDULE)]
            row = random.Random(a.seed + step).choice(pools[lang])
            ctx = context(row)
            group = [sample(row, ctx, a.temperature) for _ in range(a.group_size)]
            rewards = [reward(row["text"], text, lang) if ended else -1.0
                       for tokens, text, ended in group]
            adv = advantages(rewards)
            current = dict(tree_flatten(model.trainable_parameters()))
            old_probs = [mx.stop_gradient(log_probs(model, *ctx, tokens, a.temperature))
                         for tokens, _, _ in group]
            mx.eval(old_probs)
            model.load_weights(list(reference.items()), strict=False)
            try:
                ref_probs = [mx.stop_gradient(log_probs(model, *ctx, tokens, a.temperature))
                             for tokens, _, _ in group]
                mx.eval(ref_probs)
            finally:
                model.load_weights(list(current.items()), strict=False)
            for (tokens, _, _), old, ref, advantage in zip(group, old_probs, ref_probs, adv):
                def objective(m):
                    lp = log_probs(m, *ctx, tokens, a.temperature)
                    return grpo_loss(lp, old, ref, advantage, a.beta)
                loss, grad = nn.value_and_grad(model, objective)(model)
                mx.eval(loss, grad)
                grads = grad if grads is None else tree_map(lambda x, y: x + y, grads, grad)
                losses.append(float(loss.item()))
            grads = tree_map(lambda g: g / a.group_size, grads)
            diagnostics = {"id": row["id"], "rewards": rewards, "advantages": adv,
                           "zero_advantage": all(abs(x) < 1e-8 for x in adv),
                           "completions": [text for _, text, _ in group]}
        grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)
        mx.eval(grads, norm)
        norm_value = float(norm.item())
        if not all(math.isfinite(x) for x in losses + [norm_value]):
            raise FloatingPointError("nonfinite loss/gradient; checkpoint not advanced")
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state)
        event = {"step": step, "stage": a.stage, "loss": sum(losses) / len(losses),
                 "gradient_norm": norm_value, "seconds": time.perf_counter() - began,
                 "peak_memory_bytes": mx.get_peak_memory(), **diagnostics}
        with (a.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(event, ensure_ascii=False), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            checkpoint(step)
        mx.clear_cache()
    evaluate(f"step-{a.steps:06d}")
    result = json.loads(run_file.read_text())
    result.update(status="complete", finished=time.time(), step=a.steps,
                  peak_memory_bytes=mx.get_peak_memory())
    atomic_json(run_file, result)


if __name__ == "__main__":
    main()
