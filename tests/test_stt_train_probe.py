"""TRAIN-only greedy-versus-sampled probe through the real trainer run() with a tiny fake Qwen on the CPU."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "training/qwen_asr"
sys.path.insert(0, str(DIRECTORY))
import sampling_probe  # noqa: E402
from common import atomic_json, sentence_epoch_order, sentence_key, sha256  # noqa: E402

PRE_FEATURE = "60ef255efd7a2686be720ac28e9f0b6caecfa567"
EOS, CUT, CUT_TEXT = 0, 99, "잘린"
# reference: (greedy, sample 1, sample 2) output; None = no EOS before max tokens
SCRIPT = {"가나 다라": ("가나다라", "가나 다라", ""),
          "마바 사아": ("마바 사아", "마바 사아", None),
          "あいう": ("あいう", "あい", "あいう"),
          "かきくけ": ("かき", "かきく", "かき"),
          "你好世界": (None, "你好世界", "你好"),
          "谢谢": ("", "", ""),
          "red blue green": ("red blue", "red blue green", "red"),
          "one two": ("one two", "one", "one two")}
TEXTS = {"ko": ["가나 다라", "마바 사아"], "ja": ["あいう", "かきくけ"],
         "zh": ["你好世界", "谢谢"], "en": ["red blue green", "one two"]}


def synthetic_pools(sentences, recordings=lambda k: 1 + k % 3):
    return {lang: [{"id": f"{lang}/{k}/{r}", "language": lang, "text": f"{lang} sentence {k}",
                    "source_sentence_id": f"{lang}-{k}", "audio": f"{lang}-{k}-{r}.wav", "sha256": f"{lang}{k}{r}"}
                   for k in range(sentences) for r in range(recordings(k))]
            for lang in ("en", "ko", "ja", "zh")}


def test_full_plan_is_the_seeded_sentence_epoch_prefix_interleaved_by_language():
    pools = synthetic_pools(230)
    plan = sampling_probe.plan(pools)
    assert plan == sampling_probe.plan(pools) and len(plan) == 800
    assert [row["index"] for row in plan] == list(range(800))
    assert [row["language"] for row in plan] == ["ko", "ja", "zh", "en"] * 200
    order = sentence_epoch_order(pools, 20261030)
    for lang in ("ko", "ja", "zh", "en"):
        rows = [row for row in plan if row["language"] == lang]
        assert [row["id"] for row in rows] == [row["id"] for row in order[lang][:200]]
        assert len({row["id"] for row in rows}) == len({tuple(row["sentence"]) for row in rows}) == 200
        # Any limit is a per-language prefix of the full plan.
        for limit in (1, 2, 50):
            assert [row["id"] for row in plan[:4 * limit] if row["language"] == lang] == \
                [row["id"] for row in rows[:limit]]
    assert all("text" not in row for row in plan)
    assert plan[0]["reference_sha256"] == sampling_probe.text_sha256(order["ko"][0]["text"])
    with pytest.raises(ValueError, match="fewer than 200"):
        sampling_probe.plan(synthetic_pools(199))


def test_decode_seeds_are_stable_per_clip_and_draw():
    seeds = {(clip, draw): sampling_probe.draw_seed(clip, draw) for clip in ("ko/1/0", "ja/1/0") for draw in range(9)}
    assert len(set(seeds.values())) == 18 and all(0 <= s < 1 << 32 for s in seeds.values())
    assert seeds == {key: sampling_probe.draw_seed(*key) for key in seeds}
    spec = sampling_probe.expected(sampling_probe.plan(synthetic_pools(200)))
    assert len(spec) == 7200
    assert {(s["draw"] == 0, s["temperature"]) for s in spec.values()} == {(True, 0.0), (False, 0.7)}


def make_world(tmp_path, monkeypatch, texts, script):
    """Real run() with a fake Qwen: each decode's output is chosen by the MLX seed set just before it."""
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    optim = pytest.importorskip("mlx.optimizers")
    pytest.importorskip("jiwer")
    mx.set_default_device(mx.cpu)
    model_dir, data = tmp_path / "model", tmp_path / "data"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"frozen base")
    atomic_json(model_dir / "training-source.json",
                {"id": "mlx-community/Qwen3-ASR-1.7B-8bit", "revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"})
    protected = tmp_path / "test.jsonl"
    protected.write_text(json.dumps({"reference": "sealed test sentence", "sha256": "sealed"}) + "\n")
    (data / "wav").mkdir(parents=True)
    rows = {"train": [], "dev": []}
    for lang, sentences in texts.items():
        for k, text in enumerate(sentences):
            audio = data / f"wav/{lang}-{k}.wav"
            audio.write_text(f"{lang}/{k}")
            rows["train"].append({"id": f"{lang}/{k}", "language": lang, "text": text,
                                  "source_sentence_id": f"{lang}-{k}", "speaker": f"{lang}-{k}",
                                  "audio": f"wav/{audio.name}", "sha256": sha256(audio), "seconds": 1.0})
        audio = data / f"wav/dev-{lang}.wav"
        audio.write_text(f"dev/{lang}")
        rows["dev"].append({"id": f"dev/{lang}", "language": lang, "text": f"development only {lang}",
                            "speaker": f"dev-{lang}", "audio": f"wav/{audio.name}", "sha256": sha256(audio)})
    for split, entries in rows.items():
        (data / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in entries))
    atomic_json(data / "data.json", {"train_sha256": sha256(data / "train.jsonl"),
                                     "dev_sha256": sha256(data / "dev.jsonl"),
                                     "protected_manifest_sha256": sha256(protected)})
    train = rows["train"]
    names = [Path(r["audio"]).name for r in train]
    spy = SimpleNamespace(inputs=[], decodes=[], loads=0, fail_after=None, seeds=[])
    vocab = {}

    class Tokenizer:
        eos_token_id = EOS

        def encode(self, text, add_special_tokens=False):
            spy.inputs.append(("encode", text))
            return [1]

        def decode(self, tokens, skip_special_tokens=True):
            words = {token: text for text, token in vocab.items()}
            return CUT_TEXT if tokens and tokens[0] == CUT else words[tokens[0]] if tokens else ""

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(2, 2) for _ in range(4))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer() for _ in range(28)]

    class FakeQwen(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Body()
            self._tokenizer = Tokenizer()

        def _eos_token_ids(self):
            return [EOS]

        def _preprocess_audio(self, path):
            spy.inputs.append(("audio", path))
            return mx.zeros((1, 2)), None, Path(path).name

        def get_audio_features(self, features, mask):
            return features

        def _build_prompt(self, count, language):
            spy.inputs.append(("prompt", count, language))
            return mx.array([[1000 + names.index(count)]])

        def _build_inputs_embeds(self, prompt, audio):
            return mx.zeros((1, prompt.shape[1], 2))

    class LoRALinear(nn.Module):
        @classmethod
        def from_base(cls, linear, r, scale):
            lora = cls()
            lora.linear, lora.lora_a, lora.lora_b = linear, mx.zeros((2, r)), mx.zeros((r, 2))
            return lora

    holder = {}

    def load_model(path):
        spy.loads += 1
        holder["model"] = FakeQwen()
        return holder["model"]

    def generate_step(prompt, model, input_embeddings, max_tokens, sampler, prefill_step_size):
        if spy.fail_after is not None and len(spy.decodes) >= spy.fail_after:
            raise RuntimeError("simulated worker crash")
        row = train[int(prompt[0].item()) - 1000]
        draws = [d for d in range(sampling_probe.SAMPLES + 1)
                 if sampling_probe.draw_seed(row["id"], d) == spy.seeds[-1]]
        assert len(draws) == 1, "decode did not set its own clip/draw seed"
        draw = draws[0]
        assert sampler == (0.0 if draw == 0 else 0.7) and (max_tokens, prefill_step_size) == (256, 256)
        spy.decodes.append((row["id"], draw))
        output = script(row["text"], draw)
        tokens = [CUT] * 3 if output is None else [EOS] if output == "" else \
            [vocab.setdefault(output, len(vocab) + 1), EOS]
        for token in tokens:
            yield mx.array(token), None

    def module(name, **attributes):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
        for key, value in attributes.items():
            setattr(sys.modules[name], key, value)

    for name in ("mlx_audio", "mlx_audio.stt", "mlx_audio.lm", "mlx_lm", "mlx_lm.tuner"):
        module(name)
    module("mlx_audio.stt.utils", load_model=load_model)
    module("mlx_audio.lm.generate", generate_step=generate_step)
    module("mlx_lm.sample_utils", make_sampler=lambda temp: temp)
    module("mlx_lm.tuner.lora", LoRALinear=LoRALinear)
    real_seed = mx.random.seed
    monkeypatch.setattr(mx.random, "seed", lambda value: (spy.seeds.append(value), real_seed(value))[1])

    def forbidden(*args, **kwargs):
        raise AssertionError("the probe must not compute losses, gradients or updates")

    monkeypatch.setattr(nn, "value_and_grad", forbidden)
    monkeypatch.setattr(optim.Optimizer, "update", forbidden)
    spec = importlib.util.spec_from_file_location("qwen_train_probe", DIRECTORY / "train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    monkeypatch.setattr(trainer, "training_targets", forbidden)

    def checkpoint(name, value=0.5, train_sha256=None):
        folder = tmp_path / name
        folder.mkdir()
        weights = {f"model.layers.{i}.self_attn.{p}.{w}": mx.full((2, 16) if w == "lora_a" else (16, 2), value)
                   for i in range(28) for p in ("q_proj", "k_proj", "v_proj", "o_proj") for w in ("lora_a", "lora_b")}
        mx.save_safetensors(str(folder / "adapters.safetensors"), weights)
        model_files = {str(p.relative_to(model_dir)): sha256(p) for p in sorted(model_dir.rglob("*")) if p.is_file()}
        atomic_json(folder / "checkpoint.json", {
            "stage": "sft", "step": 1, "rank": 16, "scale": 1.0, "model_files": model_files,
            "adapters_sha256": sha256(folder / "adapters.safetensors"),
            "contract": {"data": {"train_sha256": train_sha256 or sha256(data / "train.jsonl")},
                         "settings": {"seed": 7, "sampler": "sentence-epoch", "accumulate": 4},
                         "init_adapter_sha256": None}})
        return folder

    def run(output, adapter, *extra, stage="probe"):
        monkeypatch.setattr(sys, "argv", [
            "train.py", stage, "--model", str(model_dir), "--data", str(data), "--protected-test", str(protected),
            "--output", str(tmp_path / output), "--init-adapter", str(adapter), "--memory-gb", "4.5",
            *map(str, extra)])
        a = trainer.arguments()
        a.output.mkdir(parents=True, exist_ok=True)
        trainer.run(a)
        return a.output

    return SimpleNamespace(run=run, checkpoint=checkpoint, spy=spy, holder=holder, train=train,
                           data=data, tmp=tmp_path, trainer=trainer, vocab=vocab)


@pytest.fixture
def small(tmp_path, monkeypatch):
    """Two sentences per language, one greedy plus two sampled draws per clip."""
    monkeypatch.setattr(sampling_probe, "SENTENCES", 2)
    monkeypatch.setattr(sampling_probe, "SAMPLES", 2)
    return make_world(tmp_path, monkeypatch, TEXTS, lambda reference, draw: SCRIPT[reference][draw])


def decodes(output):
    return [json.loads(line) for line in (output / "decodes.jsonl").read_text().splitlines()]


def test_native_counts_smoke_decodes_72_from_the_full_plan_and_stays_partial(tmp_path, monkeypatch):
    texts = {lang: [f"{lang} sentence {k}" for k in range(200)] for lang in ("ko", "ja", "zh", "en")}
    world = make_world(tmp_path, monkeypatch, texts, lambda reference, draw: reference)
    out = world.run("probe", world.checkpoint("adapter"), "--probe-limit", 2)
    plan = json.loads((out / "probe-plan.json").read_text())
    assert len(plan) == 800 and [row["language"] for row in plan[:8]] == ["ko", "ja", "zh", "en"] * 2
    rows = decodes(out)
    assert len(rows) == len(world.spy.decodes) == 72
    assert {(r["clip"], r["draw"]) for r in rows} == {(p["id"], d) for p in plan[:8] for d in range(9)}
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "partial"
    assert run["contract"]["probe"] == {"sentences": 200, "samples": 8, "seed": 20261030, "temperature": 0.7,
                                        "korean_wer_weight": 0.5, "language_order": ["ko", "ja", "zh", "en"],
                                        "plan_sha256": sha256(out / "probe-plan.json")}
    progress = json.loads((out / "probe-progress.json").read_text())
    assert progress["scope"].startswith("PARTIAL") and (progress["decodes"], progress["full_decodes"]) == (72, 7200)
    assert not (out / "probe-summary.json").exists()


def test_probe_sees_audio_only_updates_nothing_and_summarizes_by_hand(small):
    from mlx.utils import tree_flatten
    mx = pytest.importorskip("mlx.core")
    adapter = small.checkpoint("adapter")
    adapter_hash = sha256(adapter / "adapters.safetensors")
    out = small.run("probe", adapter)

    # No reference or development text reaches the model; only plan audio and its known language.
    assert not [value for value in small.spy.inputs if value[0] == "encode"]
    references = [text for sentences in TEXTS.values() for text in sentences]
    for value in small.spy.inputs:
        assert not any(isinstance(part, str) and any(ref in part for ref in references) for part in value)
    assert {v[1] for v in small.spy.inputs if v[0] == "audio"} == {str(small.data / r["audio"]) for r in small.train}
    parameters = dict(tree_flatten(small.holder["model"].trainable_parameters()))
    assert len(parameters) == 224 and all(bool(mx.all(v == 0.5)) for v in parameters.values())
    assert sha256(adapter / "adapters.safetensors") == adapter_hash
    assert sorted(p.name for p in out.iterdir()) == ["decodes.jsonl", "probe-plan.json", "probe-summary.json",
                                                    "run.json"]
    assert json.loads((out / "run.json").read_text())["status"] == "complete"

    rows = {(r["clip"], r["draw"]): r for r in decodes(out)}
    assert len(rows) == 24
    for (clip, draw), r in rows.items():
        assert (r["seed"], r["temperature"]) == (sampling_probe.draw_seed(clip, draw), 0.0 if draw == 0 else 0.7)
    vocab = small.vocab
    assert (rows["ko/0", 0]["tokens"], rows["ko/0", 0]["text"], rows["ko/0", 0]["reward"]) == \
        ([vocab["가나다라"], EOS], "가나다라", 0.5)  # Korean spacing costs half via the WER weight
    assert (rows["zh/0", 0]["tokens"], rows["zh/0", 0]["ended"], rows["zh/0", 0]["text"],
            rows["zh/0", 0]["reward"]) == ([CUT] * 3, False, CUT_TEXT, -1.0)
    assert (rows["zh/1", 0]["tokens"], rows["zh/1", 0]["text"], rows["zh/1", 0]["reward"]) == ([EOS], "", 0.0)

    summary = json.loads((out / "probe-summary.json").read_text())
    assert summary["decodes"] == 24 and summary["init_adapter_sha256"] == adapter_hash
    assert "not deployable" in summary["scope"] and "RL ceiling" in summary["scope"]
    expected = {  # greedy, sampled, best-of-2, headroom, errors, better, perfect,
                  # greedy empty/truncated, sampled empty/truncated, parent-drawn sentences
        "ko": (0.75, 0.25, 1.0, 0.25, 1, 1, 1, 0, 0, 1, 1, 2),
        "ja": (0.75, (2 / 3 + 1 + 0.75 + 0.5) / 4, 0.875, 0.125, 1, 1, 0, 0, 0, 0, 0, 1),
        "zh": (-0.5, 0.375, 0.5, 1.0, 2, 1, 1, 1, 1, 2, 0, 1),
        "en": (5 / 6, (1 + 1 / 3 + 0.5 + 1) / 4, 1.0, 1 / 6, 1, 1, 1, 0, 0, 0, 0, 0)}
    for lang, values in expected.items():
        got = summary["languages"][lang]
        assert (got["clips"], got["unique_clips"], got["unique_sentences"]) == (2, 2, 2)
        assert [got[k] for k in ("greedy_reward", "sampled_reward", "oracle_best_of_2_reward",
                                 "oracle_headroom")] == pytest.approx(values[:4])
        assert [got[k] for k in ("greedy_errors", "greedy_errors_with_better_sample",
                                 "greedy_errors_with_perfect_sample", "greedy_empty", "greedy_truncated",
                                 "sampled_empty", "sampled_truncated", "parent_drawn_sentences",
                                 "parent_drawn_recordings")] == [*values[4:], values[-1]]


def comparable(rows):
    return sorted((r["clip"], r["draw"], r["seed"], r["temperature"], r["tokens"], r["text"], r["ended"],
                   r["reward"]) for r in rows)


def test_limit_crash_torn_line_and_resume_match_one_uninterrupted_run(small):
    adapter = small.checkpoint("adapter")
    straight = decodes(small.run("straight", adapter))
    small.spy.decodes.clear()

    out = small.run("probe", adapter, "--probe-limit", 1)
    assert len(decodes(out)) == 12 and json.loads((out / "run.json").read_text())["status"] == "partial"
    small.spy.fail_after = 16
    with pytest.raises(RuntimeError, match="simulated"):
        small.run("probe", adapter, "--resume")
    assert len(decodes(out)) == 16 and json.loads((out / "run.json").read_text())["status"] == "running"
    with (out / "decodes.jsonl").open("a") as stream:
        stream.write('{"clip": "ja/1", "dr')  # torn final write from an evicted worker
    small.spy.fail_after = None
    small.run("probe", adapter, "--resume")
    assert len(small.spy.decodes) == len(set(small.spy.decodes)) == 24  # nothing decoded twice
    assert comparable(decodes(out)) == comparable(straight)
    assert json.loads((out / "run.json").read_text())["status"] == "complete"

    loads = small.spy.loads
    with pytest.raises(FileExistsError, match="already complete"):
        small.run("probe", adapter, "--resume")
    assert small.spy.loads == loads  # refused before loading the model


def test_changed_inputs_settings_seed_or_code_are_refused_before_loading(small, monkeypatch):
    adapter = small.checkpoint("adapter")
    out = small.run("probe", adapter, "--probe-limit", 1)
    run_file = out / "run.json"
    original = run_file.read_text()
    for change in ("adapter", "memory", "seed", "samples", "code"):
        with monkeypatch.context() as patch:
            extra = []
            if change == "adapter":
                adapter_used = small.checkpoint("other", value=0.25)
            else:
                adapter_used = adapter
            if change == "memory":
                extra = ["--memory-gb", "4.0"]
            elif change == "seed":
                patch.setattr(sampling_probe, "SEED", 20261031)
            elif change == "samples":
                patch.setattr(sampling_probe, "SAMPLES", 3)
            elif change == "code":
                run = json.loads(original)
                run["contract"]["code"]["train.py"] = "0" * 64
                run_file.write_text(json.dumps(run))
            with pytest.raises(ValueError, match="identical"):
                small.run("probe", adapter_used, "--resume", *extra)
        run_file.write_text(original)
    assert small.spy.loads == 1


@pytest.mark.parametrize("tamper,error", [
    ("edit", "drift"), ("duplicate", "drift"), ("unknown", "drift"), ("foreign", "drift"),
    ("corrupt", "corrupt"), ("plan", "changed probe plan")])
def test_edited_duplicate_foreign_or_corrupt_records_are_refused(small, tamper, error):
    adapter = small.checkpoint("adapter")
    out = small.run("probe", adapter, "--probe-limit", 1)
    path = out / "decodes.jsonl"
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])

    def redigest(row):
        row = {k: v for k, v in row.items() if k != "record_sha256"}
        return json.dumps(sampling_probe.record(
            {k: row[k] for k in ("clip", "draw", "index", "language", "reference_sha256", "temperature", "seed")},
            row["contract_sha256"], row["tokens"], row["text"], row["ended"], row["seconds"], row["reward"]),
            ensure_ascii=False)

    if tamper == "edit":
        lines[0] = json.dumps({**first, "text": "가나 다라"}, ensure_ascii=False)
    elif tamper == "duplicate":
        lines.append(lines[0])
    elif tamper == "unknown":
        lines[0] = redigest({**first, "draw": 5})
    elif tamper == "foreign":
        lines[0] = redigest({**first, "contract_sha256": "0" * 64})
    elif tamper == "corrupt":
        lines[3] = lines[3][:20]
    if tamper == "plan":
        plan = json.loads((out / "probe-plan.json").read_text())
        plan[0], plan[1] = plan[1], plan[0]
        (out / "probe-plan.json").write_text(json.dumps(plan))
    else:
        path.write_text("".join(line + "\n" for line in lines))
    with pytest.raises(ValueError, match=error):
        small.run("probe", adapter, "--resume")
    assert small.spy.loads == 1


def test_adapter_trained_on_other_data_or_changed_audio_is_refused_before_loading(small):
    with pytest.raises(ValueError, match="own training data"):
        small.run("probe", small.checkpoint("foreign", train_sha256="f" * 64))
    (small.data / small.train[0]["audio"]).write_text("changed")
    with pytest.raises(ValueError, match="changed audio"):
        small.run("probe", small.checkpoint("adapter"))
    assert small.spy.loads == 0


def parse(monkeypatch, stage, extra):
    monkeypatch.setattr(sys, "argv", ["train.py", stage, "--model", "m", "--data", "d", "--protected-test", "t",
                                      "--output", "o", *extra])
    spec = importlib.util.spec_from_file_location("qwen_train_probe_args", DIRECTORY / "train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer.arguments()


@pytest.mark.parametrize("stage,extra", [
    ("probe", []), ("probe", ["--base-reference"]),
    *[("probe", ["--init-adapter", "a", *flag]) for flag in (
        ["--seed", "20261030"], ["--seed=20261029"], ["--se=20261029"],
        ["--max-t=256"], ["--temperature", "0.7"],
        ["--temperature", "0.5"], ["--korean-wer-weight", "0.5"], ["--max-tokens", "128"],
        ["--steps", "5"], ["--learning-rate", "1e-5"], ["--eval-clips", "2"], ["--skip-initial-eval"],
        ["--external", "probe.jsonl"], ["--sampler", "sentence-epoch"], ["--max-draws", "2"],
        ["--probe-limit", "0"], ["--probe-limit", "201"])],
    ("evaluate", ["--probe-limit", "1"]), ("sft", ["--probe-limit", "1"])])
def test_incompatible_flags_are_refused(monkeypatch, stage, extra):
    with pytest.raises(SystemExit) as error:
        parse(monkeypatch, stage, extra)
    assert error.value.code == 2


def test_probe_arguments_accept_only_the_adapter_limit_and_runtime_flags(monkeypatch):
    a = parse(monkeypatch, "probe", ["--init-adapter", "a", "--probe-limit", "200", "--memory-gb", "4.5", "--resume"])
    assert (a.stage, a.probe_limit, a.memory_gb, a.resume) == ("probe", 200, 4.5, True)
    assert parse(monkeypatch, "sft", []).probe_limit is None


def pre_feature_trainer():
    shown = subprocess.run(["git", "-C", str(DIRECTORY), "show", f"{PRE_FEATURE}:training/qwen_asr/train.py"],
                           capture_output=True, text=True)
    if shown.returncode:
        pytest.skip(f"pre-feature trainer {PRE_FEATURE} unavailable")
    return shown.stdout


def test_inference_and_likelihood_paths_are_unchanged():
    def nested(source):
        run = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "run")
        return {n.name: ast.dump(n) for n in run.body if isinstance(n, ast.FunctionDef)}
    before, after = nested(pre_feature_trainer()), nested((DIRECTORY / "train.py").read_text())
    assert before == after and {"context", "sample", "log_probs", "evaluate", "load_adapter"} <= set(after)


@pytest.mark.parametrize("stage,extra", [
    ("sft", ["--steps", "2", "--learning-rate", "0.05", "--save-every", "1"]),
    ("grpo", ["--steps", "3", "--learning-rate", "0.05", "--save-every", "1"]),
    ("evaluate", [])])
def test_default_stages_match_the_pre_feature_trainer(monkeypatch, tmp_path, stage, extra):
    mx = pytest.importorskip("mlx.core")
    from test_stt_training import EQUAL, GRPO_EOS, MIXED, fake_grpo_runtime
    script = [*MIXED, *EQUAL, [2, GRPO_EOS], [1, GRPO_EOS], [3, 3, 3], [2, GRPO_EOS]] if stage == "grpo" else []
    env = fake_grpo_runtime(monkeypatch, tmp_path, script)
    previous = ModuleType("qwen_train_pre_probe")
    previous.__file__ = str(DIRECTORY / "train.py")  # same code/scorer paths
    exec(compile(pre_feature_trainer(), previous.__file__, "exec"), previous.__dict__)
    previous.training_targets = env.trainer.training_targets  # same likelihood recording
    observed = {}
    for name, trainer in (("current", env.trainer), ("base", previous)):
        env.remaining[:] = script
        del env.seeds[:], env.targets[:], env.contexts[:]
        env.configure(extra, stage=stage)
        args = trainer.arguments()
        args.output = tmp_path / name
        args.output.mkdir()
        with env.cpu():
            trainer.run(args)
        metrics = args.output / "metrics.jsonl"
        events = [json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
        for event in events:
            del event["seconds"], event["peak_memory_bytes"]
        tensors = {f"{p.name}/{kind}": mx.load(str(p / f"{kind}.safetensors"))
                   for p in sorted(args.output.glob("step-*")) for kind in ("adapters", "optimizer")}
        dev = {p.name: json.loads(p.read_text())["rows"] for p in sorted(args.output.glob("dev-*.json"))}
        contract = json.loads((args.output / "run.json").read_text())["contract"]
        observed[name] = (events, tensors, dev, contract, list(env.seeds), list(env.targets), list(env.contexts))
    current, base = observed["current"], observed["base"]
    assert current[1].keys() == base[1].keys() and bool(current[2]) and (stage == "evaluate") != bool(current[1])
    for key, arrays in current[1].items():
        assert arrays.keys() == base[1][key].keys()
        assert all(mx.array_equal(arrays[n], base[1][key][n]).item() for n in arrays), key
    assert current[0] == base[0] and current[2:] == base[2:]


def test_journal_recovers_torn_utf8_tail_and_preserves_corrupt_committed_bytes(tmp_path):
    plan = sampling_probe.plan(synthetic_pools(200))
    expected = sampling_probe.expected(plan)
    spec = expected[plan[0]['id'], 0]
    row = sampling_probe.record(spec, 'contract', [1, 0], '한글', True, 0.1, 1.0)
    committed = (json.dumps(row, ensure_ascii=False) + '\n').encode()
    torn = b'{"text":"' + '界'.encode()[:2]
    path = tmp_path / 'decodes.jsonl'
    path.write_bytes(committed + torn)
    assert sampling_probe.saved_decodes(path, expected, 'contract') == {(spec['clip'], 0): row}
    assert path.read_bytes() == committed
    damaged = committed + b'\xff\n' + torn
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match='corrupt'):
        sampling_probe.saved_decodes(path, expected, 'contract')
    assert path.read_bytes() == damaged
