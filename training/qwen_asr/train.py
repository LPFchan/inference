#!/usr/bin/env python3
"""Qwen3-ASR decoder LoRA SFT and serialized, audio-conditioned GRPO on MLX."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path
import wave

from common import (LANGUAGES, advantages, atomic_json, decayed_learning_rate, grpo_row, informative,
                    load_manifests, read_jsonl, reward, recover_uncommitted_updates, sentence_epoch_draw,
                    sentence_epoch_order, sha256, summarize, training_targets, validate_splits)
import sampling_probe

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
    p.add_argument("stage", choices=("sft", "grpo", "evaluate", "probe"))
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--protected-test", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    reference = p.add_mutually_exclusive_group()
    reference.add_argument("--init-adapter", type=Path)
    reference.add_argument("--base-reference", action="store_true",
                           help="start GRPO from the untouched ASR base and anchor KL to it")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-initial-eval", action="store_true")
    p.add_argument("--steps", type=int, default=200, help="absolute optimizer-step target")
    p.add_argument("--accumulate", type=int, default=4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--scale", type=float, default=1.0, help="LoRA multiplier (alpha/rank)")
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--lr-decay-start", type=int,
                   help="SFT only: keep the peak rate through this optimizer update, then decay linearly")
    p.add_argument("--lr-decay-end", type=int,
                   help="SFT only: last update of the decay (default --steps); a later --steps is refused")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=0.02)
    p.add_argument("--advantage-scale", type=float,
                   help="GRPO only: divide group-centered rewards by this fixed value instead of the group std")
    p.add_argument("--max-draws", type=int, default=1,
                   help="GRPO only: groups tried per update while all rewards are equal")
    p.add_argument("--reference-member", action="store_true",
                   help="GRPO only: add the TRAIN transcript to each group as an off-policy member")
    p.add_argument("--korean-wer-weight", type=float, default=0.0)
    p.add_argument("--sampler", choices=("replacement", "sentence-epoch"), default="replacement",
                   help="SFT only: draw with replacement, or each source sentence at most once")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--eval-clips", type=int, default=4,
                   help="per language on development only; per cohort with --external")
    p.add_argument("--external", type=Path,
                   help="frozen external probe manifest to transcribe instead of development")
    p.add_argument("--memory-gb", type=float, default=18.0)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--probe-limit", type=int,
                   help="probe only: stop after this many plan clips per language; --resume extends")
    a = p.parse_args()
    if a.stage == "probe":
        # The probe's counts, seed, temperature and reward are fixed in sampling_probe.
        fixed = ("steps", "accumulate", "learning_rate", "group_size", "temperature", "beta",
                 "korean_wer_weight", "max_tokens", "save_every", "eval_clips", "skip_initial_eval", "seed")
        supplied = {arg.split("=", 1)[0] for arg in sys.argv[1:]
                    if arg.startswith("--") and arg != "--"}
        changed = [name for name in fixed
                   if any(("--" + name.replace("_", "-")).startswith(flag) for flag in supplied)]
        if changed:
            p.error(f"probe uses its fixed recipe; remove {', '.join('--' + n.replace('_', '-') for n in changed)}")
        if not a.init_adapter:
            p.error("probe requires --init-adapter")
    if a.probe_limit is not None and (a.stage != "probe" or not 1 <= a.probe_limit <= sampling_probe.SENTENCES):
        p.error(f"--probe-limit must be 1..{sampling_probe.SENTENCES} and is only valid for probe")
    if min(a.steps, a.accumulate, a.rank, a.save_every, a.eval_clips, a.max_tokens, a.max_draws) < 1:
        p.error("counts must be positive")
    if a.max_draws != 1 and a.stage != "grpo":
        p.error("--max-draws is only valid for GRPO")
    if not 0 <= a.korean_wer_weight <= 1:
        p.error("Korean WER weight must be between zero and one")
    if a.group_size < 2 or a.temperature <= 0 or a.beta < 0 or a.memory_gb <= 0:
        p.error("invalid GRPO/memory settings")
    if a.advantage_scale is not None and (
            a.stage != "grpo" or not (math.isfinite(a.advantage_scale) and a.advantage_scale > 0)):
        p.error("--advantage-scale must be positive and finite, and is only valid for GRPO")
    if a.reference_member and (a.stage != "grpo" or a.max_draws != 1):
        p.error("--reference-member is only valid for GRPO with --max-draws 1")
    if a.base_reference and a.stage != "grpo":
        p.error("--base-reference is only valid for GRPO")
    if a.sampler != "replacement" and a.stage != "sft":
        p.error("--sampler sentence-epoch is only valid for SFT")
    if a.external and a.stage != "evaluate":
        p.error("--external is only valid for evaluate")
    if a.stage == "grpo" and not (a.init_adapter or a.base_reference):
        p.error("GRPO requires --init-adapter or --base-reference")
    a.learning_rate = a.learning_rate or (1e-4 if a.stage == "sft" else 1e-6)
    if a.learning_rate <= 0:
        p.error("positive learning rate required")
    if a.lr_decay_end is not None and a.lr_decay_start is None:
        p.error("--lr-decay-end requires --lr-decay-start")
    if a.lr_decay_start is not None:
        if a.stage != "sft":
            p.error("--lr-decay-start is only valid for SFT")
        a.lr_decay_end = a.steps if a.lr_decay_end is None else a.lr_decay_end
        if not 0 <= a.lr_decay_start < a.lr_decay_end or a.steps > a.lr_decay_end:
            p.error("learning-rate decay needs 0 <= start < end and --steps <= end")
    return a


def training_settings(a):
    settings = {key: getattr(a, key) for key in (
        "stage", "accumulate", "rank", "scale", "learning_rate", "group_size",
        "temperature", "beta", "max_tokens", "seed", "memory_gb", "korean_wer_weight",
        "base_reference", "advantage_scale", "sampler", "max_draws", "reference_member")}
    # --steps is only a stop target; the decay endpoint is fixed for the whole run.
    settings["lr_schedule"] = None if a.lr_decay_start is None else {
        "decay_start": a.lr_decay_start, "decay_end": a.lr_decay_end}
    return settings


def sft_row(pools, order, seed, index):
    """SFT microstep row: seeded replacement draw, or the sentence-epoch order if given."""
    if order is not None:
        return sentence_epoch_draw(order, index, SCHEDULE)
    return random.Random(seed + index).choice(pools[SCHEDULE[index % len(SCHEDULE)]])


def external_plan(manifest, per_cohort, seed):
    """Verify a frozen external probe; return its rows and a seeded, nested per-cohort subset.

    The subset for N clips per cohort is a prefix of the subset for any larger N,
    so a smoke run extends to the full probe without changing earlier clips.
    """
    manifest = Path(manifest)
    if sha256(manifest) != json.loads((manifest.parent / "probe.json").read_text())["probe_sha256"]:
        raise ValueError("changed external manifest")
    rows = read_jsonl(manifest)
    for row in rows:
        path = manifest.parent / row["audio"]
        if sha256(path) != row["sha256"]:
            raise ValueError(f"changed audio: {row['id']}")
        with wave.open(str(path)) as audio:
            layout = (audio.getnchannels(), audio.getsampwidth(), audio.getframerate())
            seconds = audio.getnframes() / audio.getframerate()
        if layout != (1, 2, 16000) or abs(seconds - row["seconds"]) > 1e-3 or not 1 <= seconds <= 15:
            raise ValueError(f"unexpected external audio format or duration: {row['id']}")
        if not row.get("cohort"):
            raise ValueError(f"missing external cohort: {row['id']}")
    plan = []
    for cohort in sorted({row["cohort"] for row in rows}):
        group = sorted((row for row in rows if row["cohort"] == cohort), key=lambda row: row["id"])
        random.Random(seed).shuffle(group)
        plan += group[:per_cohort]
    return rows, plan


def saved_predictions(path, probe, contract_sha256):
    """Completed external rows; reject anything not produced under this exact contract."""
    path = Path(path)
    text = path.read_text() if path.exists() else ""
    lines = text.splitlines()
    if text and not text.endswith("\n"):
        try:
            json.loads(lines[-1])
        except json.JSONDecodeError:
            lines.pop()  # an interrupted worker may leave its last write incomplete
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(line + "\n" for line in lines))
        temporary.replace(path)
    expected = {row["id"]: row for row in probe}
    saved = {}
    for line in lines:
        row = json.loads(line)
        source = expected.get(row["id"])
        if (row["id"] in saved or source is None or row["contract_sha256"] != contract_sha256
                or (row["reference"], row["language"], row["cohort"])
                != (source["text"], source["language"], source["cohort"])):
            raise ValueError(f"saved external prediction drift: {row['id']}")
        saved[row["id"]] = row
    return saved


def cohort_metrics(rows):
    """Shared-scorer WER/CER per language and cohort; never pooled across cohorts."""
    groups = {}
    for row in rows:
        groups.setdefault(f"{row['language']}/{row['cohort']}", []).append(row)
    return {key: {**summarize(group),
                  "empty_outputs": sum(not row["hypothesis"].strip() for row in group),
                  "truncated_outputs": sum(not row["ended"] for row in group)}
            for key, group in sorted(groups.items())}


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
    if a.external:
        a.external = a.external.resolve()
        probe, plan = external_plan(a.external, a.eval_clips, a.seed)
        # Also proves the probe shares no clip, text or speaker with this training data.
        validate_splits(train + dev, probe, read_jsonl(a.protected_test))
    pools = {lang: [r for r in train if r["language"] == lang] for lang in LANGUAGES}
    if any(not pool for pool in pools.values()):
        raise ValueError("all four training languages required")
    order = None
    if a.sampler == "sentence-epoch":
        order = sentence_epoch_order(pools, a.seed)
        for index in range(a.steps * a.accumulate):  # refuse a repeat before any model load or update
            sentence_epoch_draw(order, index, SCHEDULE)
    model_files = {str(p.relative_to(a.model)): sha256(p) for p in sorted(a.model.rglob("*"))
                   if p.is_file() and ".cache" not in p.parts and not p.name.startswith(".")}
    if not any(name.endswith(".safetensors") for name in model_files):
        raise ValueError("missing model weights")
    source = json.loads((a.model / "training-source.json").read_text())
    if source != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("unexpected base model source")
    init = a.init_adapter.resolve() if a.init_adapter else None
    if (a.external or a.stage == "probe") and init and json.loads((init / "checkpoint.json").read_text())[
            "contract"]["data"]["train_sha256"] != data_meta["train_sha256"]:
        raise ValueError("--data must be the checkpoint's own training data")
    settings = training_settings(a) if a.stage != "probe" else {
        key: getattr(a, key) for key in ("stage", "rank", "scale", "max_tokens", "memory_gb")}
    contract = {"settings": settings, "model_files": model_files,
                "data": {k: data_meta[k] for k in (
                    "train_sha256", "dev_sha256", "protected_manifest_sha256")},
                "init_adapter_sha256": sha256(init / "adapters.safetensors") if init else None,
                "code": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
                "scorer_sha256": sha256(Path(__file__).resolve().parents[2] / "scripts/stt-score.py")}
    if a.external:
        contract["external_sha256"] = sha256(a.external)
    if a.stage == "probe":
        probe_plan = sampling_probe.plan(pools)
        contract["probe"] = {**sampling_probe.recipe(),
                             "plan_sha256": hashlib.sha256(sampling_probe.plan_bytes(probe_plan)).hexdigest()}
    run_file = a.output / "run.json"
    if run_file.exists():
        if not a.resume or json.loads(run_file.read_text())["contract"] != contract:
            raise ValueError("existing run requires --resume and identical inputs/settings/code")
    elif a.resume:
        raise ValueError("no run to resume")
    else:
        atomic_json(run_file, {"contract": contract, "platform": platform.platform(),
                              "started": time.time(), "status": "running"})
    if a.external:
        contract_sha256 = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        predictions = a.output / "predictions.jsonl"
        summary = a.output / f"summary-{len(plan)}.json"
        if summary.exists():
            raise FileExistsError(f"external evaluation already complete: {summary}")
        saved = saved_predictions(predictions, probe, contract_sha256)
    if a.stage == "probe":
        contract_sha256 = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        probe_summary = a.output / "probe-summary.json"
        if probe_summary.exists():
            raise FileExistsError(f"probe already complete: {probe_summary}")
        sampling_probe.freeze_plan(a.output / "probe-plan.json", probe_plan)
        plan_decodes = sampling_probe.expected(probe_plan)
        decodes = a.output / "decodes.jsonl"
        saved = sampling_probe.saved_decodes(decodes, plan_decodes, contract_sha256)
    if a.resume:
        result = json.loads(run_file.read_text())
        result.update(status="running")
        result.pop("finished", None)
        result.pop("peak_memory_bytes", None)
        atomic_json(run_file, result)

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
        if a.stage == "grpo" and (info["stage"] not in ("sft", "grpo") or info["step"] < 1):
            raise ValueError("RL reference must be a trained SFT or GRPO checkpoint")
    # Zero-delta initialization preserves the base policy in base-reference
    # mode. The fixed starting reference shares the frozen model weights.
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
            if len(results) % 20 == 0:
                print(json.dumps({"phase": "evaluation", "tag": tag, "clips": len(results)}), flush=True)
            mx.clear_cache()
        atomic_json(a.output / f"dev-{tag}.json", {"rows": results,
                    "seconds": time.perf_counter() - started,
                    "peak_memory_bytes": mx.get_peak_memory()})

    if a.stage == "probe":
        limit = a.probe_limit or sampling_probe.SENTENCES
        references = {row["id"]: row["text"] for row in train}
        with decodes.open("a") as stream:
            for clip in probe_plan[:limit * len(sampling_probe.ORDER)]:
                pending = [plan_decodes[clip["id"], draw] for draw in range(sampling_probe.SAMPLES + 1)
                           if (clip["id"], draw) not in saved]
                if not pending:
                    continue
                # The model sees audio and the known language only; the TRAIN
                # reference scores each completion after it is decoded.
                audio_only = {"audio": clip["audio"], "language": clip["language"]}
                ctx = context(audio_only)
                for spec in pending:
                    mx.random.seed(spec["seed"])
                    began = time.perf_counter()
                    tokens, text, ended = sample(audio_only, ctx, spec["temperature"])
                    seconds = time.perf_counter() - began
                    score = reward(references[clip["id"]], text, clip["language"],
                                   sampling_probe.KOREAN_WER_WEIGHT) if ended else -1.0
                    row = sampling_probe.record(spec, contract_sha256, tokens, text, ended, seconds, score)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    saved[spec["clip"], spec["draw"]] = row
                mx.clear_cache()
        result = json.loads(run_file.read_text())
        if limit < sampling_probe.SENTENCES:
            atomic_json(a.output / "probe-progress.json",
                        {**sampling_probe.progress(limit, saved), "peak_memory_bytes": mx.get_peak_memory()})
            result.update(status="partial", finished=time.time(), peak_memory_bytes=mx.get_peak_memory())
            atomic_json(run_file, result)
            return
        parent = json.loads((init / "checkpoint.json").read_text())
        parent_draws = None
        if parent["stage"] == "sft":
            recorded = parent["contract"]["settings"]
            order = (sentence_epoch_order(pools, recorded["seed"])
                     if recorded.get("sampler") == "sentence-epoch" else None)
            parent_draws = [sft_row(pools, order, recorded["seed"], index)
                            for index in range(parent["step"] * recorded["accumulate"])]
        atomic_json(probe_summary, {
            **sampling_probe.summary(probe_plan, saved, parent_draws),
            "contract_sha256": contract_sha256, "plan_sha256": contract["probe"]["plan_sha256"],
            "init_adapter": str(init), "init_adapter_sha256": contract["init_adapter_sha256"],
            "parent": {"stage": parent["stage"], "step": parent["step"],
                       "settings": parent["contract"]["settings"],
                       "init_adapter_sha256": parent["contract"].get("init_adapter_sha256")},
            "peak_memory_bytes": mx.get_peak_memory()})
        result.update(status="complete", finished=time.time(), peak_memory_bytes=mx.get_peak_memory())
        atomic_json(run_file, result)
        return

    if a.external:
        with predictions.open("a") as stream:
            for row in plan:
                if row["id"] in saved:
                    continue
                began = time.perf_counter()
                # The model sees audio and the known language only. An absolute
                # path replaces the --data root inside context().
                audio_only = {"audio": str(a.external.parent / row["audio"]), "language": row["language"]}
                _, text, ended = sample(audio_only, context(audio_only), 0.0)
                saved[row["id"]] = {
                    "id": row["id"], "language": row["language"], "cohort": row["cohort"],
                    "reference": row["text"], "hypothesis": text, "ended": ended,
                    "seconds": row["seconds"], "batch_seconds": time.perf_counter() - began,
                    "batch_samples": 1, "contract_sha256": contract_sha256}
                stream.write(json.dumps(saved[row["id"]], ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                mx.clear_cache()
        atomic_json(summary, {
            "manifest": str(a.external), "manifest_sha256": contract["external_sha256"],
            "contract_sha256": contract_sha256, "init_adapter": str(init) if init else None,
            "clips": len(plan), "seed": a.seed,
            "scope": "balanced external probe; cohorts separate, not natural-population accuracy "
                     "or input to the primary FLEURS score",
            "cohorts": cohort_metrics([saved[row["id"]] for row in plan]),
            "peak_memory_bytes": mx.get_peak_memory()})
        result = json.loads(run_file.read_text())
        result.update(status="complete", finished=time.time(), peak_memory_bytes=mx.get_peak_memory())
        atomic_json(run_file, result)
        return

    if start == 0 and not a.skip_initial_eval and not (a.output / "dev-initial.json").exists():
        evaluate("initial")
    if a.stage == "evaluate":
        if a.skip_initial_eval:
            evaluate("initial")
        result = json.loads(run_file.read_text())
        result.update(status="complete", finished=time.time(), peak_memory_bytes=mx.get_peak_memory())
        atomic_json(run_file, result)
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
                row = sft_row(pools, order, a.seed, (step - 1) * a.accumulate + micro)
                if order is not None:
                    diagnostics.setdefault("ids", []).append(row["id"])
                ctx = context(row)
                completion = model._tokenizer.encode(row["text"], add_special_tokens=False) + [eos]
                loss, grad = nn.value_and_grad(model, lambda m: -log_probs(m, *ctx, completion).mean())(model)
                mx.eval(loss, grad)
                grads = grad if grads is None else tree_map(lambda x, y: x + y, grads, grad)
                mx.eval(grads)  # a lazy sum would keep every microstep gradient alive
                losses.append(float(loss.item()))
            grads = tree_map(lambda g: g / a.accumulate, grads)
        else:
            lang = SCHEDULE[(step - 1) % len(SCHEDULE)]
            skipped = []
            # Redraw a same-language row while all rewards are equal; only the
            # stopping group updates. The last allowed group is kept regardless.
            for attempt in range(a.max_draws):
                row = grpo_row(pools[lang], a.seed, step, attempt)
                ctx = context(row)
                group = [sample(row, ctx, a.temperature) for _ in range(a.group_size)]
                rewards = [reward(row["text"], text, lang, a.korean_wer_weight) if ended else -1.0
                           for tokens, text, ended in group]
                if informative(rewards) or attempt + 1 == a.max_draws:
                    break
                skipped.append({"id": row["id"], "rewards": rewards})
            members, member_rewards = list(group), list(rewards)
            if a.reference_member:
                # The TRAIN transcript joins the group under the unchanged reward. As an
                # off-policy member its behaviour probability is one, so its ratio is its
                # current probability; its advantage is never negative, so the clip cannot bind.
                members.append((model._tokenizer.encode(row["text"], add_special_tokens=False) + [eos],
                                row["text"], True))
                member_rewards.append(reward(row["text"], row["text"], lang, a.korean_wer_weight))
            adv = advantages(member_rewards, a.advantage_scale)
            current = dict(tree_flatten(model.trainable_parameters()))
            old_probs = [mx.stop_gradient(log_probs(model, *ctx, tokens, a.temperature))
                         for tokens, _, _ in members]
            mx.eval(old_probs)
            if a.reference_member:
                reference_probability = float(mx.exp(old_probs[-1]).mean().item())
                old_probs[-1] = mx.zeros_like(old_probs[-1])
            model.load_weights(list(reference.items()), strict=False)
            try:
                ref_probs = [mx.stop_gradient(log_probs(model, *ctx, tokens, a.temperature))
                             for tokens, _, _ in members]
                mx.eval(ref_probs)
            finally:
                model.load_weights(list(current.items()), strict=False)
            for (tokens, _, _), old, ref, advantage in zip(members, old_probs, ref_probs, adv):
                def objective(m):
                    lp = log_probs(m, *ctx, tokens, a.temperature)
                    return grpo_loss(lp, old, ref, advantage, a.beta)
                loss, grad = nn.value_and_grad(model, objective)(model)
                mx.eval(loss, grad)
                grads = grad if grads is None else tree_map(lambda x, y: x + y, grads, grad)
                mx.eval(grads)
                losses.append(float(loss.item()))
            grads = tree_map(lambda g: g / len(members), grads)
            diagnostics = {"id": row["id"], "rewards": rewards, "advantages": adv,
                           "zero_advantage": all(abs(x) < 1e-8 for x in adv),
                           "completions": [text for _, text, _ in group],
                           "draws": attempt + 1, "exhausted": not informative(rewards),
                           "skipped": skipped}
            if a.reference_member:
                diagnostics.update(reference_reward=member_rewards[-1],
                                   reference_mean_probability=reference_probability)
        grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)
        mx.eval(grads, norm)
        norm_value = float(norm.item())
        if not all(math.isfinite(x) for x in losses + [norm_value]):
            raise FloatingPointError("nonfinite loss/gradient; checkpoint not advanced")
        # From the fixed peak and absolute update index, replacing any restored rate;
        # the constant prefix leaves the optimizer's rate untouched.
        rate = decayed_learning_rate(a.learning_rate, step, settings["lr_schedule"])
        if rate is not None:
            optimizer.learning_rate = rate
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state)
        if settings["lr_schedule"] is not None:
            diagnostics["learning_rate"] = float(optimizer.learning_rate.item())
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
