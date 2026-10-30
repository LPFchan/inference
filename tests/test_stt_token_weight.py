"""Language-mean SFT token weighting and the teacher-forced TRAIN NLL probe, through the real trainer
run() with a tiny 8-bit-quantized BF16 decoder and the real mlx-lm LoRA layer on the CPU."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import wave

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "training/qwen_asr"
sys.path.insert(0, str(DIRECTORY))
import token_weight  # noqa: E402
from common import atomic_json, sentence_epoch_draw, sentence_epoch_order, sentence_key, sha256  # noqa: E402

PRE_FEATURE = "04ad123a312d1da7a32bde47da4190f780881be7"
LANGS = ("en", "ko", "ja", "zh")
CHARS = {"en": "ab", "ko": "가나", "ja": "あい", "zh": "你好"}
D, VOCAB, EOS = 64, 40, 39
SEED = 7
SCHEDULE = ("ko", "ja", "zh") * 3 + ("en",)  # train.SCHEDULE; checked in the NLL probe test


def text(lang, length):
    return (CHARS[lang] * length)[:length]


def lengths(k):
    return 2 + 3 * k  # differing transcript lengths per sentence


def write_wav(path, seconds=1.0):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * int(16000 * seconds))


def write_snapshot(directory, train, dev, protected):
    directory.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train), ("dev", dev)):
        (directory / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    atomic_json(directory / "data.json", {"train_sha256": sha256(directory / "train.jsonl"),
                                          "dev_sha256": sha256(directory / "dev.jsonl"),
                                          "protected_manifest_sha256": sha256(protected)})
    return directory


def make_world(tmp_path, monkeypatch, length=lengths, sentences=6):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    optim = pytest.importorskip("mlx.optimizers")
    pytest.importorskip("mlx_lm")
    pytest.importorskip("jiwer")
    from mlx.utils import tree_map
    mx.set_default_device(mx.cpu)
    model_dir, data = tmp_path / "model", tmp_path / "data"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"frozen base")
    atomic_json(model_dir / "training-source.json",
                {"id": "mlx-community/Qwen3-ASR-1.7B-8bit", "revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"})
    protected = tmp_path / "test.jsonl"
    protected.write_text(json.dumps({"reference": "sealed test sentence", "sha256": "sealed"}) + "\n")
    (data / "wav").mkdir(parents=True)
    train, dev = [], []
    for lang in LANGS:
        for k in range(sentences):  # FLEURS-style: one source sentence ID shared across languages, including 0
            write_wav(data / f"wav/{lang}-{k}.wav")
            train.append({"id": f"t/{lang}/{k}", "language": lang, "text": text(lang, length(k)),
                          "speaker": f"{lang}-{k}", "audio": f"wav/{lang}-{k}.wav",
                          "sha256": sha256(data / f"wav/{lang}-{k}.wav"), "seconds": 1.0, "source_sentence_id": k})
        write_wav(data / f"wav/dev-{lang}.wav", 1.5)
        dev.append({"id": f"dev/{lang}", "language": lang, "text": f"development only {lang}",
                    "speaker": f"dev-{lang}", "audio": f"wav/dev-{lang}.wav",
                    "sha256": sha256(data / f"wav/dev-{lang}.wav")})
    write_snapshot(data, train, dev, protected)
    spy = SimpleNamespace(loads=0, audio=[], encodes=0, fail_after=None, forbid=False)

    class Tokenizer:
        eos_token_id = EOS

        def encode(self, value, add_special_tokens=True):
            spy.encodes += 1
            if spy.fail_after is not None and spy.encodes > spy.fail_after:
                raise RuntimeError("simulated worker crash")
            return [1 + ord(c) % (VOCAB - 2) for c in value]  # one token per character

        def decode(self, tokens, skip_special_tokens=True):
            return ""

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(D, D, bias=False) for _ in range(4))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(VOCAB, D)
            self.layers = [Layer() for _ in range(28)]

        def __call__(self, inputs_embeds):
            h = inputs_embeds
            for layer in self.layers:
                a = layer.self_attn
                mixed = mx.cumsum(a.v_proj(h), axis=1) / mx.arange(1, h.shape[1] + 1)[None, :, None]
                h = h + 0.1 * a.o_proj(a.q_proj(h) * mx.sigmoid(a.k_proj(h)) + mixed)
            return h

    class TinyQwen(nn.Module):
        def __init__(self):
            super().__init__()
            self.model, self.lm_head = Decoder(), nn.Linear(D, VOCAB, bias=False)
            self._tokenizer = Tokenizer()

        def _eos_token_ids(self):
            return [EOS]

        def _preprocess_audio(self, path):
            spy.audio.append(path)
            return None, None, Path(path).stem

        def get_audio_features(self, features, mask):
            return mx.zeros((1, 1, D))

        def _build_prompt(self, count, language):
            marker = int(hashlib.sha256(f"{count}:{language}".encode()).hexdigest(), 16)
            return mx.array([[1, 2 + marker % 30, 3 + marker % 31]])

        def _build_inputs_embeds(self, prompt, audio):
            return self.model.embed_tokens(prompt)

    def load_model(path):
        spy.loads += 1
        model = TinyQwen()
        keys = iter(mx.random.split(mx.random.key(1234), 1000))  # a fixed base, independent of the run seed
        model.update(tree_map(lambda p: 0.05 * mx.random.normal(p.shape, key=next(keys)), model.parameters()))
        model.set_dtype(mx.bfloat16)
        nn.quantize(model.model, group_size=32, bits=8)
        spy.model = model
        return model

    def generate_step(prompt, model, input_embeddings, max_tokens, sampler, prefill_step_size):
        if spy.forbid:
            raise AssertionError("the TRAIN NLL probe must not decode")
        yield mx.array(EOS), None

    for name in ("mlx_audio", "mlx_audio.stt", "mlx_audio.lm"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", SimpleNamespace(load_model=load_model))
    monkeypatch.setitem(sys.modules, "mlx_audio.lm.generate", SimpleNamespace(generate_step=generate_step))
    monkeypatch.setattr(mx, "set_memory_limit", lambda limit: None)
    monkeypatch.setattr(mx, "set_cache_limit", lambda limit: None)
    real_value_and_grad, real_update = nn.value_and_grad, optim.Optimizer.update

    def guarded(real):
        def call(*args, **kwargs):
            if spy.forbid:
                raise AssertionError("the TRAIN NLL probe must not compute gradients or updates")
            return real(*args, **kwargs)
        return call

    monkeypatch.setattr(nn, "value_and_grad", guarded(real_value_and_grad))
    monkeypatch.setattr(optim.Optimizer, "update", guarded(real_update))
    spec = importlib.util.spec_from_file_location("qwen_train_token_weight", DIRECTORY / "train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    def argv(stage, output, *extra, trainer_module=None):
        monkeypatch.setattr(sys, "argv", [
            "train.py", stage, "--model", str(model_dir), "--data", str(data), "--protected-test", str(protected),
            "--output", str(tmp_path / output), "--memory-gb", "1", *map(str, extra)])
        a = (trainer_module or trainer).arguments()
        a.output.mkdir(parents=True, exist_ok=True)
        return a

    def run(stage, output, *extra, trainer_module=None):
        a = argv(stage, output, *extra, trainer_module=trainer_module)
        (trainer_module or trainer).run(a)
        return a.output

    def sft(output, *extra, steps=2):
        return run("sft", output, "--steps", steps, "--accumulate", 4, "--learning-rate", 0.01, "--seed", SEED,
                   "--sampler", "sentence-epoch", "--skip-initial-eval", "--eval-clips", 1, "--save-every", 1,
                   *extra)

    return SimpleNamespace(trainer=trainer, run=run, sft=sft, argv=argv, spy=spy, train=train, dev=dev,
                           data=data, protected=protected, tmp=tmp_path, model_dir=model_dir, mx=mx)


def events(output):
    return [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]


def tensors(path):
    import mlx.core as mx
    return {name: (value.dtype, value.shape, memoryview(value).tobytes())
            for name, value in mx.load(str(path)).items()}


def hand_means(rows):
    return {lang: sum(len(r["text"]) + 1 for r in rows if r["language"] == lang)
            / sum(r["language"] == lang for r in rows) for lang in LANGS}


def drawn_rows(train, steps, accumulate=4, seed=SEED):
    pools = {lang: [r for r in train if r["language"] == lang] for lang in LANGS}
    order = sentence_epoch_order(pools, seed)
    return [sentence_epoch_draw(order, i, SCHEDULE) for i in range(steps * accumulate)]


@pytest.fixture
def world(tmp_path, monkeypatch):
    return make_world(tmp_path, monkeypatch)


# ---------------------------------------------------------------- CLI

def parse(monkeypatch, stage, extra):
    spec = importlib.util.spec_from_file_location("qwen_train_token_args", DIRECTORY / "train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    monkeypatch.setattr(sys, "argv", ["train.py", stage, "--model", "m", "--data", "d", "--protected-test", "t",
                                      "--output", "o", *extra])
    return trainer, trainer.arguments()


@pytest.mark.parametrize("stage,extra", [
    ("sft", ["--sft-token-weight", "language-mean"]),
    ("sft", ["--sft-token-reference-data", "full"]),
    ("sft", ["--sft-token-weight", "off", "--sft-token-reference-data", "full"]),
    ("sft", ["--sft-token-weight", "pooled", "--sft-token-reference-data", "full"]),
    ("grpo", ["--base-reference", "--sft-token-weight", "language-mean", "--sft-token-reference-data", "full"]),
    ("evaluate", ["--sft-token-weight", "language-mean", "--sft-token-reference-data", "full"]),
    ("probe", ["--init-adapter", "a", "--sft-token-weight", "language-mean", "--sft-token-reference-data", "f"]),
    ("evaluate", ["--train-nll-probe", "plan.json"]),
    ("sft", ["--init-adapter", "a", "--train-nll-probe", "plan.json"]),
    ("probe", ["--init-adapter", "a", "--train-nll-probe", "plan.json"]),
    ("evaluate", ["--init-adapter", "a", "--train-nll-probe", "plan.json", "--external", "probe.jsonl"]),
    *[("evaluate", ["--init-adapter", "a", "--train-nll-probe", "plan.json", *flag]) for flag in (
        ["--seed", "1"], ["--se=1"], ["--eval-clips", "2"], ["--skip-initial-eval"], ["--max-tokens", "8"],
        ["--temperature", "0.5"], ["--steps", "3"], ["--probe-limit", "1"])],
])
def test_invalid_token_weight_and_nll_probe_flags_are_refused(monkeypatch, stage, extra):
    with pytest.raises(SystemExit) as error:
        parse(monkeypatch, stage, extra)
    assert error.value.code == 2


def test_default_settings_are_unchanged_and_new_flags_parse(monkeypatch):
    trainer, args = parse(monkeypatch, "sft", [])
    assert (args.sft_token_weight, args.sft_token_reference_data, args.train_nll_probe) == ("off", None, None)
    assert "sft_token_weight" not in trainer.training_settings(args)
    _, args = parse(monkeypatch, "sft", ["--sft-token-weight", "language-mean", "--sft-token-reference-data", "f"])
    assert (args.sft_token_weight, args.sft_token_reference_data) == ("language-mean", Path("f"))
    _, args = parse(monkeypatch, "evaluate", ["--init-adapter", "a", "--train-nll-probe", "p", "--memory-gb", "2",
                                              "--resume"])
    assert (args.train_nll_probe, args.init_adapter, args.memory_gb, args.resume) == (Path("p"), Path("a"), 2, True)


# ---------------------------------------------------------------- pure helpers

def test_statistics_count_transcript_tokens_plus_eos_by_hand():
    rows = [{"language": "en", "text": "abc"}, {"language": "en", "text": "abcdefg"},
            {"language": "ko", "text": "가"}, {"language": "ja", "text": "あいう"},
            {"language": "zh", "text": "你好"}, {"language": "zh", "text": "你好你"}, {"language": "zh", "text": "你"}]
    stats = token_weight.statistics(rows, lambda value: len(value) + 1, {"mode": "language-mean"}, 9)
    assert stats["languages"] == {
        "en": {"recordings": 2, "tokens": 12, "mean": 6.0}, "ko": {"recordings": 1, "tokens": 2, "mean": 2.0},
        "ja": {"recordings": 1, "tokens": 4, "mean": 4.0}, "zh": {"recordings": 3, "tokens": 9, "mean": 3.0}}
    assert stats["eos_token_id"] == 9 and stats["mode"] == "language-mean"
    with pytest.raises(ValueError, match="all four languages"):
        token_weight.statistics(rows[:3], len, {}, 9)


def test_objective_scales_each_clip_gradient_by_its_length_over_the_language_mean():
    mx = pytest.importorskip("mlx.core")
    weights = mx.array([0.3, -0.2, 0.5])
    features = mx.array([[1.0, 2.0, 0.5], [0.2, -1.0, 3.0], [2.0, 0.1, -0.4], [0.7, 0.7, 0.7]])
    nll = lambda w: mx.log1p(mx.exp(features @ w))  # four token losses
    (objective, mean), grad = mx.value_and_grad(lambda w: token_weight.language_mean_objective(nll(w), 2.5))(weights)
    _, mean_grad = mx.value_and_grad(lambda w: nll(w).mean())(weights)
    assert mean.item() == nll(weights).mean().item()  # the logged per-clip mean is the unweighted one
    assert objective.item() == pytest.approx(nll(weights).sum().item() / 2.5, rel=1e-6)
    assert mx.allclose(grad, mean_grad * 4 / 2.5, rtol=1e-6).item()
    (_, _), same = mx.value_and_grad(lambda w: token_weight.language_mean_objective(nll(w), 4.0))(weights)
    assert mx.allclose(same, mean_grad, rtol=1e-6).item()  # a clip of mean length keeps its gradient


def test_frozen_statistics_are_immutable(tmp_path):
    path = tmp_path / "token-weight.json"
    stats = {"mode": "language-mean", "languages": {lang: {"recordings": 1, "tokens": 3, "mean": 3.0}
                                                    for lang in LANGS}}
    with pytest.raises(ValueError, match="missing"):
        token_weight.freeze_statistics(path, stats, published=True)
    assert token_weight.freeze_statistics(path, stats, published=False) == {lang: 3.0 for lang in LANGS}
    written = path.read_bytes()
    assert token_weight.freeze_statistics(path, stats, published=True) == {lang: 3.0 for lang in LANGS}
    changed = json.loads(json.dumps(stats))
    changed["languages"]["zh"]["mean"] = 3.0000001
    with pytest.raises(ValueError, match="differ"):
        token_weight.freeze_statistics(path, changed, published=True)
    path.write_text("{")
    with pytest.raises(ValueError, match="corrupt"):
        token_weight.freeze_statistics(path, stats, published=False)
    path.write_bytes(written)


# ---------------------------------------------------------------- SFT

def test_language_mean_weights_differing_lengths_and_logs_the_unweighted_loss(world):
    off = world.sft("off")
    on = world.sft("on", "--sft-token-weight", "language-mean", "--sft-token-reference-data", world.data)
    means = hand_means(world.train)
    stats = json.loads((on / "token-weight.json").read_text())
    assert {lang: v["mean"] for lang, v in stats["languages"].items()} == means
    assert stats["reference_train_sha256"] == sha256(world.data / "train.jsonl")
    assert stats["reference_data_sha256"] == sha256(world.data / "data.json") and stats["eos_token_id"] == EOS
    settings = json.loads((on / "run.json").read_text())["contract"]["settings"]
    assert settings["sft_token_weight"] == {"mode": "language-mean",
                                            "reference_train_sha256": stats["reference_train_sha256"],
                                            "reference_data_sha256": stats["reference_data_sha256"]}
    assert "sft_token_weight" not in json.loads((off / "run.json").read_text())["contract"]["settings"]
    rows = {r["id"]: r for r in world.train}
    plain, weighted = events(off), events(on)
    assert [e["ids"] for e in plain] == [e["ids"] for e in weighted]
    for p, w in zip(plain, weighted):
        assert set(w) - set(p) == {"tokens", "token_normalizers", "objectives", "objective", "sequence_losses"}
        assert set(p) == {"step", "stage", "loss", "gradient_norm", "seconds", "peak_memory_bytes", "ids"}
        assert w["tokens"] == [len(rows[i]["text"]) + 1 for i in w["ids"]]
        assert w["token_normalizers"] == [means[rows[i]["language"]] for i in w["ids"]]
        assert w["loss"] == sum(w["sequence_losses"]) / 4
        assert w["objectives"] == pytest.approx([loss * t / m for loss, t, m in zip(
            w["sequence_losses"], w["tokens"], w["token_normalizers"])], rel=1e-5)
        assert w["objective"] == sum(w["objectives"]) / 4
    # The same forward and pre-update weights: the first update's logged loss is identical.
    assert weighted[0]["loss"] == plain[0]["loss"]
    assert weighted[0]["gradient_norm"] != pytest.approx(plain[0]["gradient_norm"], rel=1e-3)
    assert weighted[1]["loss"] != plain[1]["loss"]
    assert tensors(on / "step-000001/adapters.safetensors") != tensors(off / "step-000001/adapters.safetensors")


def test_equal_lengths_reproduce_the_default_update(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch, length=lambda k: 5)
    off = world.sft("off", steps=1)
    on = world.sft("on", "--sft-token-weight", "language-mean", "--sft-token-reference-data", world.data, steps=1)
    plain, weighted = events(off)[0], events(on)[0]
    assert weighted["token_normalizers"] == weighted["tokens"] == [6] * 4
    assert weighted["loss"] == plain["loss"] and weighted["objective"] == pytest.approx(plain["loss"], rel=1e-6)
    assert weighted["gradient_norm"] == pytest.approx(plain["gradient_norm"], rel=1e-5)
    mx = world.mx
    for kind in ("adapters", "optimizer"):
        a, b = (mx.load(str(run / f"step-000001/{kind}.safetensors")) for run in (on, off))
        assert a.keys() == b.keys() and all(mx.allclose(a[n], b[n], rtol=1e-4, atol=1e-7).item() for n in a)


def subset(world, name, keep, edit=None, dev=None):
    rows = [dict(r) for r in world.train if keep(r)]
    if edit:
        edit(rows)
    directory = world.tmp / name
    shutil.copytree(world.data / "wav", directory / "wav")
    return write_snapshot(directory, rows, dev or world.dev, world.protected), rows


def test_subset_preflight_keeps_full_reference_denominators(world):
    small, rows = subset(world, "longest", lambda r: int(r["id"].rsplit("/", 1)[1]) >= 4)
    a = world.argv("sft", "pre", "--steps", 1, "--accumulate", 4, "--seed", SEED, "--sampler", "sentence-epoch",
                   "--skip-initial-eval", "--eval-clips", 1, "--sft-token-weight", "language-mean",
                   "--sft-token-reference-data", world.data)
    a.data = small
    world.trainer.run(a)
    out = a.output
    stats = json.loads((out / "token-weight.json").read_text())
    assert {lang: v["mean"] for lang, v in stats["languages"].items()} == hand_means(world.train)
    assert hand_means(world.train) != hand_means(rows)
    contract = json.loads((out / "run.json").read_text())["contract"]
    assert contract["data"]["train_sha256"] == sha256(small / "train.jsonl")
    assert contract["settings"]["sft_token_weight"]["reference_train_sha256"] == sha256(world.data / "train.jsonl")
    assert contract["settings"]["sft_token_weight"]["reference_data_sha256"] == sha256(world.data / "data.json")
    event = events(out)[0]
    assert event["token_normalizers"] == [hand_means(world.train)[r["language"]]
                                          for r in ({x["id"]: x for x in rows}[i] for i in event["ids"])]


@pytest.mark.parametrize("case,error", [
    ("text", "not in the reference"), ("seconds", "not in the reference"), ("foreign", "not in the reference"),
    ("dev", "different dev_sha256"), ("drift", "changed train manifest"), ("audio", "changed audio")])
def test_invalid_subsets_and_reference_drift_are_refused_before_model_load(world, case, error):
    reference = world.data
    if case in ("drift", "audio"):
        reference = world.tmp / "reference"
        shutil.copytree(world.data, reference)
        if case == "drift":
            with (reference / "train.jsonl").open("a") as stream:
                stream.write("\n")
        else:
            write_wav(reference / world.train[0]["audio"], 2.0)
        small = world.data
    elif case == "dev":
        changed_dev = [{**world.dev[0], "text": "another development line"}, *world.dev[1:]]
        small, _ = subset(world, "sub", lambda r: True, dev=changed_dev)
    else:
        def edit(rows):
            if case == "text":
                rows[0]["text"] += "a"
            elif case == "seconds":
                rows[0]["seconds"] = 1.25
            else:
                rows[0]["id"] = "t/en/elsewhere"
        small, _ = subset(world, "sub", lambda r: True, edit=edit)
    a = world.argv("sft", "bad", "--steps", 1, "--sampler", "sentence-epoch", "--seed", SEED,
                   "--sft-token-weight", "language-mean", "--sft-token-reference-data", reference)
    a.data = small
    with pytest.raises(ValueError, match=error):
        world.trainer.run(a)
    assert world.spy.loads == 0 and not (a.output / "run.json").exists()


def test_statistics_are_frozen_and_checked_on_resume(world):
    flags = ("--sft-token-weight", "language-mean", "--sft-token-reference-data", world.data)
    out = world.sft("on", *flags, steps=1)
    path = out / "token-weight.json"
    original = path.read_bytes()
    world.sft("on", *flags, "--resume", steps=2)
    assert path.read_bytes() == original and [e["step"] for e in events(out)] == [1, 2]
    metrics = (out / "metrics.jsonl").read_bytes()
    stats = json.loads(original)
    stats["languages"]["zh"]["mean"] += 1e-9
    for content, error in ((json.dumps(stats), "differ"), ("{", "corrupt"), (None, "missing")):
        if content is None:
            path.unlink()
        else:
            path.write_text(content)
        with pytest.raises(ValueError, match=error):
            world.sft("on", *flags, "--resume", steps=3)
        assert not (out / "step-000003").exists() and (out / "metrics.jsonl").read_bytes() == metrics
    path.write_bytes(original)
    loads = world.spy.loads
    other = world.tmp / "other-reference"
    shutil.copytree(world.data, other)
    atomic_json(other / "data.json", {**json.loads((world.data / "data.json").read_text()), "note": "copy"})
    with pytest.raises(ValueError, match="identical inputs"):
        world.sft("on", "--sft-token-weight", "language-mean", "--sft-token-reference-data", other, "--resume",
                  steps=3)
    assert world.spy.loads == loads  # a changed reference binding is refused before loading
    world.sft("on", *flags, "--resume", steps=3)
    assert [e["step"] for e in events(out)] == [1, 2, 3]


def pre_feature_trainer():
    shown = subprocess.run(["git", "-C", str(DIRECTORY), "show", f"{PRE_FEATURE}:training/qwen_asr/train.py"],
                           capture_output=True, text=True)
    if shown.returncode:
        pytest.skip(f"pre-feature trainer {PRE_FEATURE} unavailable")
    previous = ModuleType("qwen_train_pre_token_weight")
    previous.__file__ = str(DIRECTORY / "train.py")  # same code/scorer paths, so the same contract
    exec(compile(shown.stdout, previous.__file__, "exec"), previous.__dict__)
    return previous


@pytest.mark.parametrize("explicit", [False, True])
def test_default_off_sft_matches_the_pre_feature_trainer(world, explicit):
    previous = pre_feature_trainer()
    base = world.run("sft", "base", "--steps", 2, "--accumulate", 4, "--learning-rate", 0.01, "--seed", SEED,
                     "--sampler", "sentence-epoch", "--skip-initial-eval", "--eval-clips", 1, "--save-every", 1,
                     trainer_module=previous)
    current = world.sft("current", *(["--sft-token-weight", "off"] if explicit else []))
    strip = lambda rows: [{k: v for k, v in e.items() if k not in ("seconds", "peak_memory_bytes")} for e in rows]
    assert strip(events(current)) == strip(events(base))
    for step in ("step-000001", "step-000002"):
        for kind in ("adapters", "optimizer"):
            assert tensors(current / step / f"{kind}.safetensors") == tensors(base / step / f"{kind}.safetensors")
    assert sorted(p.name for p in current.iterdir()) == sorted(p.name for p in base.iterdir())
    contract = lambda out: json.loads((out / "run.json").read_text())["contract"]
    assert contract(current) == contract(base)
    rows = lambda out: json.loads((out / "dev-step-000002.json").read_text())["rows"]
    assert rows(current) == rows(base)


# ---------------------------------------------------------------- TRAIN NLL probe

def freeze_plan(world, path, steps, seed=SEED, cap=256, edit=None):
    """The packet tool's rule: sentences the seed's run never drew in that language, one recording each."""
    drawn = {(r["language"], sentence_key(r)) for r in drawn_rows(world.train, steps, seed=seed)}
    probe = {lang: {"ids": [r["id"] for r in world.train if r["language"] == lang
                            and (lang, sentence_key(r)) not in drawn][:cap]} for lang in LANGS}
    if edit:
        edit(probe)
    ids = [i for lang in LANGS for i in probe[lang]["ids"]]
    path.write_text(json.dumps({"train_sha256": sha256(world.data / "train.jsonl"), "cap": cap, "probes": {
        str(seed): {"languages": probe, "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}}}))
    return path


def expected_nll(world, row):
    """Independent teacher-forced NLL with an explicit log-softmax over the full sequence."""
    mx = world.mx
    model = world.spy.model
    prompt = model._build_prompt(Path(row["audio"]).stem, {"en": "English", "ko": "Korean", "ja": "Japanese",
                                                           "zh": "Chinese"}[row["language"]])
    tokens = model._tokenizer.encode(row["text"]) + [EOS]
    embeds = mx.concatenate([model.model.embed_tokens(prompt),
                             model.model.embed_tokens(mx.array([tokens[:-1]]))], axis=1)
    logits = model.lm_head(model.model(inputs_embeds=embeds)).astype(mx.float32)[0, prompt.shape[1] - 1:]
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return len(tokens), -sum(logp[i, t].item() for i, t in enumerate(tokens))


@pytest.fixture
def trained(world):
    adapter = world.sft("twin") / "step-000002"
    plan = freeze_plan(world, world.tmp / "train-probe.json", steps=2)
    return SimpleNamespace(world=world, adapter=adapter, plan=plan)


def nll_probe(world, output, adapter, plan, *extra):
    world.spy.forbid = True
    try:
        return world.run("evaluate", output, "--init-adapter", adapter, "--train-nll-probe", plan, *extra)
    finally:
        world.spy.forbid = False


def test_nll_probe_teacher_forces_train_only_and_counts_by_hand(trained):
    world, adapter = trained.world, trained.adapter
    assert world.trainer.SCHEDULE == SCHEDULE
    adapter_hash = sha256(adapter / "adapters.safetensors")
    world.spy.audio.clear()
    out = nll_probe(world, "probe", adapter, trained.plan)
    assert sorted(p.name for p in out.iterdir()) == ["run.json", "train-nll-summary.json", "train-nll.jsonl"]
    assert sha256(adapter / "adapters.safetensors") == adapter_hash
    rows = {r["id"]: r for r in world.train}
    frozen = json.loads(trained.plan.read_text())["probes"][str(SEED)]["languages"]
    ids = [i for lang in LANGS for i in frozen[lang]["ids"]]
    # Two updates of four draws: KO 3, JA 3, ZH 2 sentences drawn; every EN sentence is unused.
    assert [len(frozen[lang]["ids"]) for lang in LANGS] == [6, 3, 3, 4]
    assert world.spy.audio == [str(world.data / rows[i]["audio"]) for i in ids]  # no development audio
    records = [json.loads(line) for line in (out / "train-nll.jsonl").read_text().splitlines()]
    assert [r["id"] for r in records] == ids
    drawn = drawn_rows(world.train, 2)
    for record in records:
        tokens, nll = expected_nll(world, rows[record["id"]])
        assert record["tokens"] == tokens == len(rows[record["id"]]["text"]) + 1
        assert record["nll_sum"] == pytest.approx(nll, rel=2e-3)
    summary = json.loads((out / "train-nll-summary.json").read_text())
    assert summary["contract_sha256"] == token_weight.contract_sha256(summary["contract"])
    assert summary["contract"]["init_adapter_sha256"] == adapter_hash
    assert summary["contract"]["settings"] == {"stage": "evaluate", "rank": 16, "scale": 1.0, "memory_gb": 1.0}
    assert summary["contract"]["train_nll_probe"] == {
        "plan_sha256": sha256(trained.plan), "seed": SEED,
        "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "checkpoint": {"step": 2, "accumulate": 4, "sampler": "sentence-epoch"}}
    assert summary["checkpoint"]["settings"]["seed"] == SEED
    for lang in LANGS:
        mine = [r for r in records if r["language"] == lang]
        got = summary["languages"][lang]
        assert (got["clips"], got["tokens"]) == (len(mine), sum(r["tokens"] for r in mine))
        assert got["token_pooled_nll"] == pytest.approx(sum(r["nll_sum"] for r in mine) / got["tokens"], rel=1e-12)
        # Shared FLEURS source IDs (0 included): a sentence drawn in another language is reported, not hidden.
        exposure = {r["id"]: sorted({d["language"] for d in drawn if d["source_sentence_id"]
                                     == rows[r["id"]]["source_sentence_id"]} - {lang}) for r in mine}
        assert {r["id"]: r["drawn_in_other_languages"] for r in summary["records"] if r["language"] == lang} \
            == exposure
        assert got["clips_drawn_in_another_language"] == sum(bool(v) for v in exposure.values())
    assert any(rows[i]["source_sentence_id"] == 0 for i in ids)
    assert summary["languages"]["en"]["clips_drawn_in_another_language"] > 0
    assert json.loads((out / "run.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("case,error", [
    ("drawn", "drawn, or repeated"), ("unknown", "not a zh TRAIN row"), ("bucket", "not a zh TRAIN row"),
    ("hash", "changed or duplicated"), ("seed", "no four-language entry"), ("duration", "duration"),
    ("replacement", "fresh sentence-epoch"), ("parent", "fresh sentence-epoch"), ("foreign", "own training data"),
    ("plan-train", "other TRAIN data")])
def test_nll_probe_refuses_unsafe_plans_and_checkpoints_before_loading(trained, case, error):
    world, adapter, plan = trained.world, trained.adapter, trained.plan
    drawn = drawn_rows(world.train, 2)
    if case == "drawn":  # a ZH sentence the run drew in ZH (source sentence ID 0 included)
        zh = next(r["id"] for r in drawn if r["language"] == "zh")
        plan = freeze_plan(world, world.tmp / "p.json", 2, edit=lambda p: p["zh"]["ids"].append(zh))
    elif case in ("unknown", "bucket"):
        clip = "t/zh/missing" if case == "unknown" else next(r["id"] for r in drawn if r["language"] == "ja")
        plan = freeze_plan(world, world.tmp / "p.json", 2, edit=lambda p: p["zh"]["ids"].append(clip))
    elif case == "hash":
        frozen = json.loads(plan.read_text())
        frozen["probes"][str(SEED)]["languages"]["en"]["ids"].reverse()
        plan.write_text(json.dumps(frozen))
    elif case == "seed":
        plan = freeze_plan(world, world.tmp / "p.json", 2, seed=SEED + 1)
    elif case == "plan-train":
        frozen = json.loads(plan.read_text())
        frozen["train_sha256"] = "0" * 64
        plan.write_text(json.dumps(frozen))
    elif case == "duration":
        rows = {r["id"]: r for r in world.train}
        clip = json.loads(plan.read_text())["probes"][str(SEED)]["languages"]["zh"]["ids"][0]
        write_wav(world.data / rows[clip]["audio"], 1.25)  # unlike its recorded 1.0 s
        rows[clip]["sha256"] = sha256(world.data / rows[clip]["audio"])
        write_snapshot(world.data, list(rows.values()), world.dev, world.protected)
        info = json.loads((adapter / "checkpoint.json").read_text())
        info["contract"]["data"]["train_sha256"] = sha256(world.data / "train.jsonl")
        atomic_json(adapter / "checkpoint.json", info)
        world.train[:] = list(rows.values())
        plan = freeze_plan(world, world.tmp / "p.json", 2)
    else:
        info = json.loads((adapter / "checkpoint.json").read_text())
        if case == "replacement":
            info["contract"]["settings"]["sampler"] = "replacement"
        elif case == "parent":
            info["contract"]["init_adapter_sha256"] = "f" * 64
        else:
            info["contract"]["data"]["train_sha256"] = "f" * 64
        atomic_json(adapter / "checkpoint.json", info)
    loads = world.spy.loads
    with pytest.raises(ValueError, match=error):
        nll_probe(world, "refused", adapter, plan)
    assert world.spy.loads == loads and not (world.tmp / "refused/run.json").exists()


def test_nll_probe_crash_torn_line_resume_and_refusals(trained):
    world, adapter, plan = trained.world, trained.adapter, trained.plan
    straight = nll_probe(world, "straight", adapter, plan)
    reference = (straight / "train-nll.jsonl").read_text().splitlines()
    world.spy.encodes, world.spy.fail_after = 0, 5
    with pytest.raises(RuntimeError, match="simulated"):
        nll_probe(world, "probe", adapter, plan)
    out = world.tmp / "probe"
    assert len((out / "train-nll.jsonl").read_text().splitlines()) == 5
    with (out / "train-nll.jsonl").open("a") as stream:
        stream.write('{"id": "t/zh/5", "tok')  # torn final write from an evicted worker
    world.spy.fail_after = None
    world.spy.encodes = 0
    nll_probe(world, "probe", adapter, plan, "--resume")
    assert world.spy.encodes == len(reference) - 5  # nothing scored twice
    resumed = (out / "train-nll.jsonl").read_text().splitlines()
    assert [json.loads(r) for r in resumed] == [json.loads(r) for r in reference]
    summary = (out / "train-nll-summary.json").read_bytes()
    loads = world.spy.loads
    with pytest.raises(FileExistsError, match="already complete"):
        nll_probe(world, "probe", adapter, plan, "--resume")
    assert world.spy.loads == loads and (out / "train-nll-summary.json").read_bytes() == summary

    partial = world.tmp / "partial"
    world.spy.encodes, world.spy.fail_after = 0, 3
    with pytest.raises(RuntimeError):
        nll_probe(world, "partial", adapter, plan)
    world.spy.fail_after = None
    lines = (partial / "train-nll.jsonl").read_text().splitlines()
    other = world.sft("other", "--sft-token-weight", "language-mean", "--sft-token-reference-data", world.data)
    with pytest.raises(ValueError, match="identical inputs"):
        nll_probe(world, "partial", other / "step-000002", plan, "--resume")
    for tamper in ("edit", "duplicate", "foreign", "corrupt"):
        changed = list(lines)
        first = json.loads(changed[0])
        if tamper == "edit":
            changed[0] = json.dumps({**first, "nll_sum": first["nll_sum"] + 1})
        elif tamper == "duplicate":
            changed.append(changed[0])
        elif tamper == "foreign":
            body = {k: v for k, v in first.items() if k != "record_sha256"}
            clip = {**body, "drawn_in_other_languages": []}
            changed[0] = json.dumps(token_weight.record(clip, "0" * 64, body["tokens"], body["nll_sum"]))
        else:
            changed[1] = changed[1][:10]
        (partial / "train-nll.jsonl").write_text("".join(line + "\n" for line in changed))
        loads = world.spy.loads
        with pytest.raises(ValueError, match="drift|corrupt"):
            nll_probe(world, "partial", adapter, plan, "--resume")
        assert world.spy.loads == loads


def test_compare_reads_two_finished_summaries_and_never_overwrites(trained, monkeypatch, capsys):
    world, plan = trained.world, trained.plan
    candidate = world.sft("candidate", "--sft-token-weight", "language-mean", "--sft-token-reference-data",
                          world.data) / "step-000002"
    reports = [nll_probe(world, name, adapter, plan) / "train-nll-summary.json"
               for name, adapter in (("probe-candidate", candidate), ("probe-twin", trained.adapter))]
    result_path = world.tmp / "comparison.json"
    monkeypatch.setattr(sys, "argv", ["token_weight.py", "compare", *map(str, reports), "--output", str(result_path)])
    capsys.readouterr()  # drop the trainer's own event lines
    token_weight.main()
    result = json.loads(result_path.read_text())
    assert json.loads(capsys.readouterr().out) == result
    summaries = [json.loads(p.read_text()) for p in reports]
    for lang in LANGS:
        got = result["languages"][lang]
        c, t = (s["languages"][lang] for s in summaries)
        assert got["clips"] == c["clips"]
        if c["clips"]:
            assert got["difference"] == pytest.approx(c["token_pooled_nll"] - t["token_pooled_nll"], rel=1e-9)
    assert result["verdict"] in ("supported", "falsified", "inconclusive")
    assert result["primary"] == "zh" and result["diagnostic"] == ["en"] and result["report_only"] == ["ko", "ja"]
    assert result["candidate"]["checkpoint"]["settings"]["sft_token_weight"]["mode"] == "language-mean"
    with pytest.raises(FileExistsError):
        token_weight.main()
    with pytest.raises(ValueError, match="same adapter"):
        token_weight.compare(summaries[1], summaries[1])
    changed = json.loads(json.dumps(summaries[0]))
    changed["contract"]["code"]["train.py"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        token_weight.compare(changed, summaries[1])
    changed["contract_sha256"] = token_weight.contract_sha256(changed["contract"])
    with pytest.raises(ValueError, match="more than the adapter"):
        token_weight.compare(changed, summaries[1])
    changed = json.loads(json.dumps(summaries[0]))
    changed["contract"]["settings"]["memory_gb"] = 2.0
    changed["contract_sha256"] = token_weight.contract_sha256(changed["contract"])
    with pytest.raises(ValueError, match="more than the adapter"):
        token_weight.compare(changed, summaries[1])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_nll_is_refused_before_a_record_can_be_published(value):
    with pytest.raises(FloatingPointError, match="nonfinite TRAIN NLL"):
        token_weight.record({}, "contract", 1, value)


def synthetic(adapter, offsets, tokens):
    contract = {"init_adapter_sha256": adapter, "train_nll_probe": {"seed": 1}}
    records = [{"id": f"{lang}/{i}", "language": lang, "tokens": t, "nll_sum": t * (1.0 + offsets[lang](i)),
                "drawn_in_other_languages": []} for lang in LANGS for i, t in enumerate(tokens[lang])]
    return {"contract": contract, "contract_sha256": token_weight.contract_sha256(contract), "records": records}


@pytest.mark.parametrize("shift,verdict", [(-0.05, "supported"), (0.0, "falsified"), (0.02, "falsified"),
                                           (None, "inconclusive")])
def test_paired_bootstrap_verdicts_thirds_and_small_cohorts(shift, verdict):
    import random
    tokens = {"zh": [5 + i % 40 for i in range(256)], "en": [10] * 256, "ko": [7] * 11, "ja": []}
    noise = random.Random(3)
    jitter = [noise.uniform(-0.2, 0.2) for _ in range(256)]
    zh = (lambda i: shift) if shift is not None else (lambda i: jitter[i] - 0.004)
    twin = synthetic("twin", {lang: (lambda i: 0.0) for lang in LANGS}, tokens)
    candidate = synthetic("cand", {"zh": zh, "en": lambda i: -0.01, "ko": lambda i: 0.0, "ja": lambda i: 0.0},
                          tokens)
    result = token_weight.compare(candidate, twin)
    assert result["verdict"] == verdict
    assert result["bootstrap"] == {"resamples": 400, "seed": 20261030, "unit": "clip, paired"}
    zh_result = result["languages"]["zh"]
    expected = (sum(t * zh(i) for i, t in enumerate(tokens["zh"]))) / sum(tokens["zh"])
    assert zh_result["difference"] == pytest.approx(expected, abs=1e-12)
    assert zh_result["ci95"][0] <= zh_result["difference"] + 1e-12
    assert result["languages"]["ja"] == {"clips": 0, "tokens": 0, "difference": None, "ci95": None}
    assert result["languages"]["ko"]["clips"] == 11 and result["languages"]["ko"]["ci95"] is not None
    thirds = result["zh_thirds_by_tokens"]
    assert thirds["shortest"]["clips"] == thirds["longest"]["clips"] == 85
    assert thirds["shortest"]["tokens"] < thirds["longest"]["tokens"]
    assert token_weight.compare(candidate, twin) == result  # deterministic seeded resampling
    one = synthetic("cand", {lang: (lambda i: -0.1) for lang in LANGS},
                    {"zh": [4], "en": [], "ko": [], "ja": []})
    lone = token_weight.compare(one, synthetic("twin", {lang: (lambda i: 0.0) for lang in LANGS},
                                               {"zh": [4], "en": [], "ko": [], "ja": []}))
    assert lone["verdict"] == "untestable" and lone["zh_thirds_by_tokens"] is None
    assert math.isclose(lone["languages"]["zh"]["difference"], -0.1)
