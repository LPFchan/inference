"""Decoder LoRA placement through the real trainer and mlx-lm LoRALinear on a tiny 8-bit BF16 Qwen (CPU)."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import wave

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm.tuner.lora")
pytest.importorskip("jiwer")
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.tuner.lora import LoRALinear  # noqa: E402

DIRECTORY = Path(__file__).resolve().parents[1] / "training/qwen_asr"
sys.path.insert(0, str(DIRECTORY))
from common import atomic_json, sha256  # noqa: E402

PREFEATURE = "4815ca601554dca68d94adf204ce9161fd71aa5d"  # trainer before --lora-targets
HIDDEN, FFN, VOCAB, EOS, AUDIO = 64, 128, 16, 0, 4
ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
TEXT = {"train": {"en": ["big cat", "red sun", "old map"], "ko": ["가방", "나무", "다리"],
                  "ja": ["さくら", "やま", "かわ"], "zh": ["中文", "山水", "日月"]},
        "dev": {"en": ["new bird"], "ko": ["바다"], "ja": ["うみ"], "zh": ["海洋"]}}


def adapter_names(projections):
    return {f"model.layers.{i}.{'self_attn' if p in ATTENTION else 'mlp'}.{p}.{w}"
            for i in range(28) for p in projections for w in ("lora_a", "lora_b")}


class Attention(nn.Module):
    def __init__(self, out, bias):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(HIDDEN, HIDDEN, bias=bias) for _ in range(3))
        setattr(self, out, nn.Linear(HIDDEN, HIDDEN, bias=bias))
        self.out_name = out  # a plain string, not a second reference to the projection

    def __call__(self, x):
        return getattr(self, self.out_name)(self.q_proj(x) * mx.sigmoid(self.k_proj(x)) + self.v_proj(x))


class MLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj, self.up_proj = nn.Linear(HIDDEN, FFN, bias=False), nn.Linear(HIDDEN, FFN, bias=False)
        self.down_proj = nn.Linear(FFN, HIDDEN, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, out="o_proj", bias=False):
        super().__init__()
        self.norm = nn.RMSNorm(HIDDEN)
        self.self_attn, self.mlp = Attention(out, bias), MLPBlock()

    def __call__(self, h):
        h = h + self.self_attn(self.norm(h))
        return h + self.mlp(self.norm(h))


class Encoder(nn.Module):
    """Same projection names as the decoder, so only the `model.layers.` prefix keeps it frozen."""

    def __init__(self):
        super().__init__()
        self.layers = [Block("out_proj", bias=True) for _ in range(2)]

    def __call__(self, h):
        for layer in self.layers:
            h = layer(h)
        return h


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = [Block() for _ in range(28)]

    def __call__(self, inputs_embeds):
        positions = mx.arange(1, inputs_embeds.shape[1] + 1).astype(inputs_embeds.dtype)
        h = mx.cumsum(inputs_embeds, axis=1) / positions[None, :, None]  # causal context
        for layer in self.layers:
            h = layer(h)
        return h


class Tokenizer:
    eos_token_id = EOS

    def __init__(self, spy):
        self.spy = spy

    def encode(self, text, add_special_tokens=True):
        self.spy.encoded.append(text)
        return [ord(c) % (VOCAB - 1) + 1 for c in text.replace(" ", "")][:3]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join("abcdefghijklmno"[t - 1] for t in tokens if t)


class TinyQwen(nn.Module):
    """Qwen3-ASR `_model` layout: audio_tower, model.layers.N.{self_attn,mlp}, lm_head."""

    def __init__(self, spy):
        super().__init__()
        self.audio_tower, self.model = Encoder(), Decoder()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self._tokenizer, self._spy = Tokenizer(spy), spy

    def _eos_token_ids(self):
        return [EOS]

    def _preprocess_audio(self, path):
        self._spy.audio.append(path)
        with wave.open(path) as audio:
            level = int.from_bytes(audio.readframes(1), "little", signed=True)
        return mx.full((1, AUDIO, HIDDEN), level / 1000, dtype=mx.bfloat16), None, AUDIO

    def get_audio_features(self, features, mask):
        return self.audio_tower(features)

    def _build_prompt(self, count, language):
        self._spy.languages.append(language)
        return mx.array([[1] + [2] * count + [3 + len(language) % 5]])

    def _build_inputs_embeds(self, prompt, audio):
        embeds = self.model.embed_tokens(prompt)
        return mx.concatenate([embeds[:, :1], audio.astype(embeds.dtype), embeds[:, 1 + audio.shape[1]:]], axis=1)


def tiny_qwen(spy, weights=None):
    model = TinyQwen(spy)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=8)
    if weights:
        model.load_weights(str(weights))
    mx.eval(model.parameters())
    return model


def generate_step(prompt, model, input_embeddings, max_tokens, sampler, prefill_step_size):
    model._spy.decodes.append((max_tokens, prefill_step_size))
    embeds = input_embeddings[None]
    for _ in range(max_tokens):
        logits = model.lm_head(model.model(inputs_embeds=embeds)[:, -1]).astype(mx.float32)
        token = sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True))[0]
        yield token, None
        embeds = mx.concatenate([embeds, model.model.embed_tokens(token.reshape(1, 1))], axis=1)


def write_wav(path, level, seconds=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(level.to_bytes(2, "little", signed=True) * int(16000 * seconds))


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


@pytest.fixture
def world(tmp_path, monkeypatch):
    spy = SimpleNamespace(encoded=[], audio=[], languages=[], decodes=[], loads=0)
    model_dir, data, probe = (tmp_path / name for name in ("model", "data", "probe"))
    model_dir.mkdir()
    with mx.stream(mx.cpu):
        mx.random.seed(5)
        mx.save_safetensors(str(model_dir / "model.safetensors"),
                            dict(tree_flatten(tiny_qwen(spy).parameters())))
    atomic_json(model_dir / "training-source.json", {"id": "mlx-community/Qwen3-ASR-1.7B-8bit",
                                                     "revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"})
    protected = tmp_path / "test.jsonl"
    write_jsonl(protected, [{"reference": "sealed test sentence", "sha256": "sealed"}])
    meta = {"protected_manifest_sha256": sha256(protected)}
    clips = (f"{split}-{lang}-{index}" for split in TEXT for lang in TEXT[split]
             for index in range(len(TEXT[split][lang])))
    for level, name in enumerate(clips, 100):
        write_wav(data / f"{name}.wav", 37 * level)  # each clip is distinct audio
    for split in TEXT:
        rows = [{"id": name, "language": lang, "text": text, "speaker": name, "audio": f"{name}.wav",
                 "sha256": sha256(data / f"{name}.wav"), "seconds": 1.0}
                for lang, texts in TEXT[split].items()
                for index, text in enumerate(texts) for name in [f"{split}-{lang}-{index}"]]
        write_jsonl(data / f"{split}.jsonl", rows)
        meta[f"{split}_sha256"] = sha256(data / f"{split}.jsonl")
    atomic_json(data / "data.json", meta)
    entries = []
    for index, (cohort, text) in enumerate((("digit", "3월 1일"), ("digit", "총 2개"),
                                            ("other", "오늘은 맑음"), ("other", "정말 좋은 하루"))):
        write_wav(probe / f"wav/{index}.wav", 900 + 41 * index, 1.5)
        entries.append({"id": f"kss/{index}", "language": "ko", "speaker": "kss/single-speaker",
                        "audio": f"wav/{index}.wav", "text": text, "seconds": 1.5, "cohort": cohort,
                        "sha256": sha256(probe / f"wav/{index}.wav")})
    write_jsonl(probe / "probe.jsonl", entries)
    atomic_json(probe / "probe.json", {"probe_sha256": sha256(probe / "probe.jsonl")})

    holder = {}

    def load_model(path):
        spy.loads += 1
        holder["model"] = tiny_qwen(spy, Path(path) / "model.safetensors")
        return holder["model"]

    for name in ("mlx_audio", "mlx_audio.stt", "mlx_audio.lm"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", SimpleNamespace(load_model=load_model))
    monkeypatch.setitem(sys.modules, "mlx_audio.lm.generate", SimpleNamespace(generate_step=generate_step))
    trainer = importlib.util.module_from_spec(
        spec := importlib.util.spec_from_file_location("qwen_train_lora_targets", DIRECTORY / "train.py"))
    spec.loader.exec_module(trainer)

    def configure(stage, output, *extra, module=None):
        monkeypatch.setattr(sys, "argv", [
            "train.py", stage, "--model", str(model_dir), "--data", str(data), "--protected-test", str(protected),
            "--output", str(tmp_path / output), "--memory-gb", "4.5", "--max-tokens", "2", "--seed", "11",
            *map(str, extra)])
        a = (module or trainer).arguments()
        a.output.mkdir(parents=True, exist_ok=True)
        return a

    def run(stage, output, *extra, module=None):
        a = configure(stage, output, *extra, module=module)
        with mx.stream(mx.cpu):
            (module or trainer).run(a)
        return a.output

    def sft(output, *extra):
        return run("sft", output, "--steps", 2, "--accumulate", 2, "--save-every", 1, "--skip-initial-eval",
                   "--eval-clips", 1, "--learning-rate", 0.01, *extra)

    return SimpleNamespace(trainer=trainer, run=run, sft=sft, spy=spy, holder=holder, model=model_dir,
                           probe=probe / "probe.jsonl", tmp=tmp_path,
                           load=lambda: tiny_qwen(spy, model_dir / "model.safetensors"))


def tensors(path):
    return {name: (value.dtype, value.shape, memoryview(value).tobytes())
            for name, value in mx.load(str(path)).items()}


def test_placement_keeps_attention_initialization_base_weights_and_zero_delta_outputs(world):
    built = {}
    for targets in ("attention", "attention-mlp"):
        with mx.stream(mx.cpu):
            mx.random.seed(11)
            model = world.load()
            base = {name: (v.dtype, memoryview(v).tobytes()) for name, v in tree_flatten(model.parameters())}
            features = mx.random.normal((1, AUDIO, HIDDEN)).astype(mx.bfloat16)
            embeds = model._build_inputs_embeds(mx.array([[1, 2, 2, 2, 2, 5, 6]]), model.get_audio_features(
                features, None))
            before = model.lm_head(model.model(inputs_embeds=embeds))
            world.trainer.add_adapters(model, targets, 16, 1.0)
            after = model.lm_head(model.model(inputs_embeds=model._build_inputs_embeds(
                mx.array([[1, 2, 2, 2, 2, 5, 6]]), model.get_audio_features(features, None))))
        assert before.dtype == after.dtype == mx.bfloat16 and mx.array_equal(before, after).item()
        wrapped = [name for name, layer in model.named_modules() if isinstance(layer, LoRALinear)]
        assert all(isinstance(layer.linear, nn.QuantizedLinear) for name, layer in model.named_modules()
                   if isinstance(layer, LoRALinear))
        assert len(wrapped) == {"attention": 112, "attention-mlp": 196}[targets]
        assert not [name for name in wrapped if not name.startswith("model.layers.")]  # encoder untouched
        trainable = dict(tree_flatten(model.trainable_parameters()))
        assert set(trainable) == adapter_names(ATTENTION + (MLP if targets == "attention-mlp" else ()))
        frozen = {name.replace(".linear.", "."): (v.dtype, memoryview(v).tobytes())
                  for name, v in tree_flatten(model.parameters()) if name not in trainable}
        assert frozen == base  # every base and audio weight is frozen and unchanged
        built[targets] = trainable
    attention, expanded = built["attention"], built["attention-mlp"]
    assert (len(attention), len(expanded)) == (224, 392)
    for name, value in attention.items():
        assert value.dtype == expanded[name].dtype == mx.float32
        assert memoryview(value).tobytes() == memoryview(expanded[name]).tobytes(), name
    for name in set(expanded) - set(attention):
        assert expanded[name].shape == ((HIDDEN if "down" not in name else FFN, 16) if name.endswith("lora_a")
                                        else (16, FFN if "down" not in name else HIDDEN))
        assert name.endswith("lora_a") or not mx.any(expanded[name]).item()


def test_wrong_decoder_geometry_is_refused(world):
    with mx.stream(mx.cpu):
        model = world.load()
        model.model.layers[3].mlp = nn.Identity()  # 81 MLP projections
        with pytest.raises(ValueError, match="expected 84 decoder MLP projections, got 81"):
            world.trainer.add_adapters(model, "attention-mlp", 16, 1.0)


def parse(monkeypatch, stage, *extra):
    spec = importlib.util.spec_from_file_location("qwen_train_targets_args", DIRECTORY / "train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    monkeypatch.setattr(sys, "argv", ["train.py", stage, "--model", "m", "--data", "d", "--protected-test", "t",
                                     "--output", "o", *extra])
    return trainer, trainer.arguments()


@pytest.mark.parametrize("stage,extra", [("sft", []), ("evaluate", []), ("evaluate", ["--external", "p.jsonl"]),
                                         ("grpo", ["--base-reference"]), ("grpo", ["--init-adapter", "c"])])
def test_placement_defaults_to_attention_is_bound_to_settings_and_refused_for_grpo(monkeypatch, stage, extra):
    trainer, args = parse(monkeypatch, stage, *extra)
    assert trainer.training_settings(args)["lora_targets"] == "attention"
    trainer, args = parse(monkeypatch, stage, *extra, "--lora-targets", "attention")
    assert trainer.training_settings(args)["lora_targets"] == "attention"
    if stage == "grpo":
        with pytest.raises(SystemExit) as error:
            parse(monkeypatch, stage, *extra, "--lora-targets", "attention-mlp")
        assert error.value.code == 2
    else:
        trainer, args = parse(monkeypatch, stage, *extra, "--lora-targets", "attention-mlp")
        assert trainer.training_settings(args)["lora_targets"] == "attention-mlp"
    for invalid in ("mlp", "all"):
        with pytest.raises(SystemExit):
            parse(monkeypatch, stage, *extra, "--lora-targets", invalid)


def rewrite(world, source, name, change=None, info_change=None):
    """Copy a checkpoint with edited tensors or metadata and a matching adapter hash."""
    folder = world.tmp / name
    folder.mkdir()
    weights = mx.load(str(source / "adapters.safetensors"))
    if change:
        change(weights)
    mx.save_safetensors(str(folder / "adapters.safetensors"), weights)
    info = json.loads((source / "checkpoint.json").read_text())
    info["adapters_sha256"] = sha256(folder / "adapters.safetensors")
    if info_change:
        info_change(info)
    atomic_json(folder / "checkpoint.json", info)
    return folder


def test_checkpoint_geometry_binds_resume_sft_and_evaluate(world, monkeypatch):
    mlp_run = world.sft("mlp", "--lora-targets", "attention-mlp")
    attention_run = world.sft("attention")
    mlp, attention = mlp_run / "step-000002", attention_run / "step-000002"
    info = json.loads((mlp / "checkpoint.json").read_text())
    assert info["lora_targets"] == info["contract"]["settings"]["lora_targets"] == "attention-mlp"
    assert json.loads((attention / "checkpoint.json").read_text())["lora_targets"] == "attention"
    weights = mx.load(str(mlp / "adapters.safetensors"))
    assert set(weights) == adapter_names(ATTENTION + MLP)
    assert all(mx.any(weights[f"model.layers.{i}.mlp.{p}.lora_b"]).item() for i in range(28) for p in MLP)
    assert len(mx.load(str(mlp / "optimizer.safetensors"))) > len(mx.load(str(attention / "optimizer.safetensors")))

    # A changed placement is refused by the resume contract before the model loads.
    loads = world.spy.loads
    for output, flags in (("mlp", []), ("attention", ["--lora-targets", "attention-mlp"])):
        with pytest.raises(ValueError, match="identical inputs/settings/code"):
            world.sft(output, "--steps", 3, "--resume", *flags)
    assert world.spy.loads == loads
    world.sft("mlp", "--steps", 3, "--resume", "--lora-targets", "attention-mlp")
    assert json.loads((mlp_run / "latest.json").read_text())["step"] == 3

    legacy = rewrite(world, attention, "legacy", info_change=lambda i: i.pop("lora_targets"))
    relabeled = rewrite(world, mlp, "relabeled", info_change=lambda i: i.update(lora_targets="attention"))
    missing = rewrite(world, mlp, "missing", lambda w: w.pop("model.layers.27.mlp.down_proj.lora_b"))
    reshaped = rewrite(world, mlp, "reshaped", lambda w: w.update({
        "model.layers.5.mlp.up_proj.lora_b": w["model.layers.5.mlp.up_proj.lora_b"][:, :HIDDEN]}))
    accepted = [(mlp, "attention-mlp"), (attention, "attention"), (legacy, "attention")]
    refused = [(mlp, "attention", "geometry"), (attention, "attention-mlp", "geometry"),
               (legacy, "attention-mlp", "geometry"), (relabeled, "attention", "invalid adapter"),
               (missing, "attention-mlp", "invalid adapter"), (reshaped, "attention-mlp", "invalid adapter")]
    for index, (checkpoint, targets) in enumerate(accepted):
        out = world.run("evaluate", f"accepted-{index}", "--init-adapter", checkpoint, "--lora-targets", targets,
                        "--skip-initial-eval", "--eval-clips", 1)
        assert len(json.loads((out / "dev-initial.json").read_text())["rows"]) == 4
    for index, (checkpoint, targets, error) in enumerate(refused):
        decodes = len(world.spy.decodes)
        with pytest.raises(ValueError, match=error):
            world.run("evaluate", f"refused-{index}", "--init-adapter", checkpoint, "--lora-targets", targets,
                      "--skip-initial-eval", "--eval-clips", 1)
        assert len(world.spy.decodes) == decodes  # refused before any sampling
        with pytest.raises(ValueError, match=error):  # and before an SFT update
            world.sft(f"refused-sft-{index}", "--init-adapter", checkpoint, "--lora-targets", targets)
        assert not (world.tmp / f"refused-sft-{index}/metrics.jsonl").exists()
    # A trained MLP adapter cannot seed attention-only GRPO either.
    with pytest.raises(ValueError, match="geometry"):
        world.run("grpo", "grpo-from-mlp", "--init-adapter", mlp, "--steps", 1, "--skip-initial-eval")


def test_external_probe_uses_the_selected_geometry_without_references(world):
    mlp = world.sft("mlp", "--lora-targets", "attention-mlp") / "step-000002"
    # Attention deltas removed and MLP deltas enlarged: only the MLP adapters can change decoding.
    mlp_only = rewrite(world, mlp, "mlp-only", lambda w: w.update({
        name: (value * 0 if ".self_attn." in name else value * 200)
        for name, value in w.items() if name.endswith("lora_b")}))
    world.spy.encoded.clear()
    world.spy.audio.clear()
    external = ["--external", world.probe, "--eval-clips", 2]
    runs = {"base-attention": world.run("evaluate", "base-attention", *external),
            "base-mlp": world.run("evaluate", "base-mlp", *external, "--lora-targets", "attention-mlp"),
            "mlp-only": world.run("evaluate", "mlp-only-eval", *external, "--lora-targets", "attention-mlp",
                                  "--init-adapter", mlp_only)}
    assert not world.spy.encoded  # probe references never reach the tokenizer or model
    assert set(world.spy.audio) == {str(world.probe.parent / f"wav/{i}.wav") for i in range(4)}
    assert world.spy.decodes[-12:] == [(2, 256)] * 12  # same greedy decode limits for every run
    contracts = {name: json.loads((out / "run.json").read_text())["contract"] for name, out in runs.items()}
    hypotheses = {name: [json.loads(line)["hypothesis"] for line in (out / "predictions.jsonl").read_text().splitlines()]
                  for name, out in runs.items()}
    # Zero-delta MLP adapters reproduce the attention-only base, so the new matched base is the old base.
    assert hypotheses["base-mlp"] == hypotheses["base-attention"]
    assert {**contracts["base-mlp"], "settings": {**contracts["base-mlp"]["settings"], "lora_targets": "attention"}} \
        == contracts["base-attention"]
    # The candidate's contract differs from its matched base only by the adapter hash.
    assert {**contracts["mlp-only"], "init_adapter_sha256": None} == contracts["base-mlp"]
    assert hypotheses["mlp-only"] != hypotheses["base-mlp"]


def test_default_path_matches_the_prefeature_trainer(world, monkeypatch):
    """No flag: SFT and GRPO reproduce the pre-feature trainer's metrics, tensors and transcripts."""
    shown = subprocess.run(["git", "-C", str(DIRECTORY), "show", f"{PREFEATURE}:training/qwen_asr/train.py"],
                           capture_output=True, text=True)
    if shown.returncode:
        pytest.skip(f"pre-feature trainer {PREFEATURE} unavailable")
    previous = ModuleType("qwen_train_prefeature")
    previous.__file__ = str(DIRECTORY / "train.py")  # same code/scorer paths
    exec(compile(shown.stdout, previous.__file__, "exec"), previous.__dict__)
    recipes = {"sft": ["--steps", 2, "--accumulate", 2, "--save-every", 1, "--eval-clips", 1,
                       "--learning-rate", 0.01],
               "grpo": ["--base-reference", "--steps", 2, "--group-size", 2, "--save-every", 1,
                        "--skip-initial-eval", "--eval-clips", 1, "--learning-rate", 0.01]}
    for stage, extra in recipes.items():
        observed = {}
        for name, module in (("current", None), ("prefeature", previous)):
            out = world.run(stage, f"{stage}-{name}", *extra, module=module)
            events = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
            for event in events:
                del event["seconds"], event["peak_memory_bytes"]
            dev = {path.name: json.loads(path.read_text())["rows"] for path in sorted(out.glob("dev-*.json"))}
            saved = {f"{step}/{kind}": tensors(out / f"step-{step:06d}/{kind}.safetensors")
                     for step in (1, 2) for kind in ("adapters", "optimizer")}
            observed[name] = (events, dev, saved, json.loads((out / "run.json").read_text())["contract"])
        (events, dev, saved, contract), (base_events, base_dev, base_saved, base_contract) = observed.values()
        assert events == base_events and dev == base_dev, stage
        assert saved == base_saved, stage
        assert len(saved["1/adapters"]) == 224
        # The only intentional contract change: placement is now recorded explicitly.
        assert contract["settings"].pop("lora_targets") == "attention"
        assert contract == base_contract
