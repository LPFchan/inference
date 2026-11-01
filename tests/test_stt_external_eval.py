"""External-probe evaluation through the real trainer run() with a tiny fake Qwen on the CPU."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import wave

import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402

DIRECTORY = Path(__file__).resolve().parents[1] / "training/qwen_asr"
sys.path.insert(0, str(DIRECTORY))
from common import atomic_json, sha256  # noqa: E402

# reference, model output; None = no EOS before max tokens
DIGIT = {"d1": ("비행기는 5분 후에 도착", "비행기는 오분 후에 도착"),
         "d2": ("3월 1일", "3월 1일"),
         "d3": ("총 2개", "")}
OTHER = {"o1": ("안녕하세요 반갑습니다", "안녕하세요 반갑습니다"),
         "o2": ("정말 좋은 하루", "정말좋은 하루"),
         "o3": ("오늘은 맑음", None)}
TRUNCATED = "오늘은"
LANGUAGE_TEXT = {"en": "plain english words", "ko": "학습 문장", "ja": "日本語の文", "zh": "中文句子"}


def write_wav(path, level, seconds=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(level.to_bytes(2, "little", signed=True) * int(16000 * seconds))


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def write_probe(probe, rows):
    write_jsonl(probe / "probe.jsonl", rows)
    atomic_json(probe / "probe.json", {"probe_sha256": sha256(probe / "probe.jsonl")})


class Spy:
    """Every value the fake model receives, plus decode and load counts."""

    def __init__(self):
        self.inputs, self.decoded, self.loads, self.fail_after = [], [], 0, None


class Tokenizer:
    eos_token_id = 0

    def __init__(self, spy, outputs):
        self.spy, self.outputs = spy, outputs

    def encode(self, text, add_special_tokens=False):
        self.spy.inputs.append(("encode", text))
        return [1]

    def decode(self, tokens, skip_special_tokens=True):
        return TRUNCATED if tokens == [-1] else self.outputs[tokens[0] - 1] if tokens else ""


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
    """Prompt token = which probe WAV was heard; output is scripted per WAV."""

    def __init__(self, spy, wavs, outputs):
        super().__init__()
        self.model = Body()
        self._tokenizer = Tokenizer(spy, outputs)
        self._wavs, self._spy = wavs, spy

    def _eos_token_ids(self):
        return [0]

    def _preprocess_audio(self, path):
        self._spy.inputs.append(("audio", path))
        return mx.zeros((1, 2)), None, Path(path).name

    def get_audio_features(self, features, mask):
        return features

    def _build_prompt(self, count, language):
        self._spy.inputs.append(("prompt", count, language))
        return mx.array([[self._wavs.index(count) + 1]])

    def _build_inputs_embeds(self, prompt, audio):
        return mx.zeros((1, prompt.shape[1], 2))


class LoRALinear(nn.Module):
    @classmethod
    def from_base(cls, linear, r, scale):
        lora = cls()
        lora.linear, lora.lora_a, lora.lora_b = linear, mx.zeros((2, r)), mx.zeros((r, 2))
        return lora


@pytest.fixture
def world(tmp_path, monkeypatch):
    mx.set_default_device(mx.cpu)
    rows = [(cohort, key, *pair) for cohort, group in (("digit", DIGIT), ("other", OTHER))
            for key, pair in group.items()]
    model, data, probe = (tmp_path / name for name in ("model", "data", "probe"))
    model.mkdir()
    (model / "weights.safetensors").write_bytes(b"frozen base")
    atomic_json(model / "training-source.json",
                {"id": "mlx-community/Qwen3-ASR-1.7B-8bit", "revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"})
    protected = tmp_path / "test.jsonl"
    write_jsonl(protected, [{"reference": "sealed test sentence", "sha256": "sealed"}])
    manifests = {}
    for level, split in enumerate(("train", "dev")):
        entries = []
        for offset, (lang, text) in enumerate(LANGUAGE_TEXT.items()):
            write_wav(data / f"wav/{split}-{lang}.wav", 1000 + 10 * level + offset)
            entries.append({"id": f"{split}-{lang}", "language": lang, "text": f"{text} {split}",
                            "speaker": f"{split}-{lang}", "audio": f"wav/{split}-{lang}.wav",
                            "sha256": sha256(data / f"wav/{split}-{lang}.wav"), "seconds": 1.0})
        write_jsonl(data / f"{split}.jsonl", entries)
        manifests[f"{split}_sha256"] = sha256(data / f"{split}.jsonl")
    atomic_json(data / "data.json", {**manifests, "protected_manifest_sha256": sha256(protected)})
    entries = []
    for index, (cohort, key, reference, _) in enumerate(rows):
        write_wav(probe / f"wav/{key}.wav", 200 + index, 1.5)
        entries.append({"id": f"kss/{key}", "language": "ko", "speaker": "kss/single-speaker",
                        "audio": f"wav/{key}.wav", "text": reference, "seconds": 1.5,
                        "sha256": sha256(probe / f"wav/{key}.wav"), "cohort": cohort})
    write_probe(probe, entries)

    spy = Spy()
    wavs = [f"{key}.wav" for _, key, _, _ in rows]
    outputs = [output for *_, output in rows]
    holder = {}

    def load_model(path):
        spy.loads += 1
        holder["model"] = FakeQwen(spy, wavs, outputs)
        return holder["model"]

    def generate_step(prompt, model, input_embeddings, max_tokens, sampler, prefill_step_size):
        if spy.fail_after is not None and len(spy.decoded) >= spy.fail_after:
            raise RuntimeError("simulated worker crash")
        index = int(prompt[0].item()) - 1
        spy.decoded.append(wavs[index])
        if outputs[index] is None:
            yield mx.array(-1), None  # stops at max tokens without EOS
            return
        if outputs[index]:
            yield mx.array(index + 1), None
        yield mx.array(0), None

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
    spec = importlib.util.spec_from_file_location("qwen_train_external", DIRECTORY / "train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    def run(output, *extra):
        monkeypatch.setattr(sys, "argv", [
            "train.py", "evaluate", "--model", str(model), "--data", str(data),
            "--protected-test", str(protected), "--output", str(tmp_path / output),
            "--external", str(probe / "probe.jsonl"), "--memory-gb", "4.5", *map(str, extra)])
        a = trainer.arguments()
        a.output.mkdir(parents=True, exist_ok=True)
        trainer.run(a)
        return tmp_path / output

    def checkpoint(name, rank=16, train_sha256=None, value=0.5):
        folder = tmp_path / name
        folder.mkdir()
        weights = {f"model.layers.{i}.self_attn.{p}.{w}": mx.full((2, rank) if w == "lora_a" else (rank, 2), value)
                   for i in range(28) for p in ("q_proj", "k_proj", "v_proj", "o_proj") for w in ("lora_a", "lora_b")}
        mx.save_safetensors(str(folder / "adapters.safetensors"), weights)
        model_files = {str(p.relative_to(model)): sha256(p) for p in sorted(model.rglob("*")) if p.is_file()}
        atomic_json(folder / "checkpoint.json", {
            "stage": "sft", "step": 10, "rank": rank, "scale": 1.0, "model_files": model_files,
            "adapters_sha256": sha256(folder / "adapters.safetensors"),
            "contract": {"data": {"train_sha256": train_sha256 or manifests["train_sha256"]}}})
        return folder

    return SimpleNamespace(run=run, checkpoint=checkpoint, spy=spy, holder=holder, probe=probe,
                           data=data, tmp=tmp_path, references=[r[2] for r in rows])


def predictions(output):
    return [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]


def test_external_probe_sees_audio_only_updates_nothing_and_scores_cohorts_separately(world):
    from mlx.utils import tree_flatten
    adapter = world.checkpoint("sft-step")
    adapter_hash = sha256(adapter / "adapters.safetensors")
    out = world.run("eval-sft", "--init-adapter", adapter, "--eval-clips", 100000)

    # No reference or development text reaches the model; nothing is teacher-forced.
    assert not [value for value in world.spy.inputs if value[0] == "encode"]
    for value in world.spy.inputs:
        assert not any(isinstance(part, str) and any(ref in part for ref in world.references)
                       for part in value)
    assert {value[2] for value in world.spy.inputs if value[0] == "prompt"} == {"Korean"}
    # The checkpoint was applied and never moved; no training or development artifacts appear.
    parameters = dict(tree_flatten(world.holder["model"].trainable_parameters()))
    assert len(parameters) == 224 and all(bool(mx.all(v == 0.5)) for v in parameters.values())
    assert sha256(adapter / "adapters.safetensors") == adapter_hash
    assert sorted(p.name for p in out.iterdir()) == ["predictions.jsonl", "run.json", "summary-6.json"]
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "complete"
    assert run["contract"]["external_sha256"] == sha256(world.probe / "probe.jsonl")
    assert run["contract"]["init_adapter_sha256"] == adapter_hash

    cohorts = json.loads((out / "summary-6.json").read_text())["cohorts"]
    assert set(cohorts) == {"ko/digit", "ko/other"}
    digit, other = cohorts["ko/digit"], cohorts["ko/other"]
    # Hand-counted: digit 1 sub + 2 deletions of 8 words, 1 sub + 3 deletions of 17 characters.
    assert digit["wer"] == pytest.approx(3 / 8) and digit["cer"] == pytest.approx(4 / 17)
    # Spacing costs words not characters; the truncated clip drops one word and two characters.
    assert other["wer"] == pytest.approx(3 / 7) and other["cer"] == pytest.approx(2 / 21)
    assert (digit["empty_outputs"], digit["truncated_outputs"]) == (1, 0)
    assert (other["empty_outputs"], other["truncated_outputs"]) == (0, 1)


def test_resume_keeps_saved_rows_and_extends_the_same_seeded_subset(world):
    smoke = predictions(world.run("eval-base", "--eval-clips", 1))
    assert sorted(r["cohort"] for r in smoke) == ["digit", "other"]
    other_model = predictions(world.run("eval-other", "--eval-clips", 1))
    assert [r["id"] for r in other_model] == [r["id"] for r in smoke]  # shared subset across models

    world.spy.decoded.clear()
    world.spy.fail_after = 2
    with pytest.raises(RuntimeError, match="simulated"):
        world.run("eval-base", "--eval-clips", 3, "--resume")
    out = world.tmp / "eval-base"
    interrupted = json.loads((out / "run.json").read_text())
    assert interrupted["status"] == "running"
    assert "finished" not in interrupted and "peak_memory_bytes" not in interrupted
    assert len(predictions(out)) == 4
    with (out / "predictions.jsonl").open("a") as stream:
        stream.write('{"id": "kss/o')  # torn final write from an evicted worker
    world.spy.fail_after = None
    world.run("eval-base", "--eval-clips", 3, "--resume")
    rows = predictions(out)
    assert len(rows) == len({r["id"] for r in rows}) == 6
    new = {Path(r["id"]).name + ".wav" for r in rows} - {Path(r["id"]).name + ".wav" for r in smoke}
    # o3 never ends, so it is decoded once more with the adapter off.
    assert sorted(world.spy.decoded) == sorted([*new, *(new & {"o3.wav"})])
    assert rows[:2] == smoke
    assert json.loads((out / "summary-2.json").read_text())["clips"] == 2
    assert json.loads((out / "summary-6.json").read_text())["clips"] == 6
    assert json.loads((out / "run.json").read_text())["status"] == "complete"

    loads = world.spy.loads
    with pytest.raises(FileExistsError, match="already complete"):
        world.run("eval-base", "--eval-clips", 3, "--resume")
    assert world.spy.loads == loads  # refused before loading the model


def rewrite_probe(world, change):
    rows = [json.loads(line) for line in (world.probe / "probe.jsonl").read_text().splitlines()]
    change(rows)
    write_probe(world.probe, rows)


@pytest.mark.parametrize("case,error", [
    ("audio", "changed audio"),
    ("manifest", "changed external manifest"),
    ("duration", "duration"),
    ("training_overlap", "overlap"),
    ("test_overlap", "test .*overlap"),
])
def test_changed_or_leaky_probe_is_rejected_before_loading(world, case, error):
    if case == "audio":
        write_wav(world.probe / "wav/d1.wav", 9999, 1.5)
    elif case == "manifest":
        with (world.probe / "probe.jsonl").open("a") as stream:
            stream.write("\n")
    elif case == "duration":
        rewrite_probe(world, lambda rows: rows[0].update(seconds=2.5))
    elif case == "training_overlap":
        rewrite_probe(world, lambda rows: rows[0].update(text=f"{LANGUAGE_TEXT['ko']} train"))
    else:
        rewrite_probe(world, lambda rows: rows[0].update(text="sealed test sentence"))
    with pytest.raises(ValueError, match=error):
        world.run("eval", "--eval-clips", 1)
    assert world.spy.loads == 0


def test_contract_drift_and_foreign_rows_are_rejected(world):
    first = world.checkpoint("first")
    world.run("eval", "--init-adapter", first, "--eval-clips", 1)
    for extra in (["--init-adapter", world.checkpoint("second", value=0.25)],
                  ["--init-adapter", first, "--max-tokens", 128]):
        with pytest.raises(ValueError, match="identical"):
            world.run("eval", *extra, "--eval-clips", 3, "--resume")
    path = world.tmp / "eval/predictions.jsonl"
    original = path.read_text()
    for tamper in ({"reference": "다른 문장"}, {"contract_sha256": "0" * 64}):
        row = {**json.loads(original.splitlines()[0]), **tamper}
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n" + original.split("\n", 1)[1])
        with pytest.raises(ValueError, match="drift"):
            world.run("eval", "--init-adapter", first, "--eval-clips", 3, "--resume")
    path.write_text(original + original.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="drift"):
        world.run("eval", "--init-adapter", first, "--eval-clips", 3, "--resume")
    assert world.spy.loads == 1


@pytest.mark.parametrize("kind,error", [("rank", "geometry"), ("data", "own training data")])
def test_checkpoint_must_match_model_geometry_and_its_training_data(world, kind, error):
    adapter = world.checkpoint("bad", rank=8) if kind == "rank" else world.checkpoint("bad", train_sha256="f" * 64)
    with pytest.raises(ValueError, match=error):
        world.run("eval", "--init-adapter", adapter, "--eval-clips", 1)
    assert not (world.tmp / "eval/predictions.jsonl").exists()
    assert (kind == "rank") == (world.tmp / "eval/run.json").exists()  # data mismatch refused before any record


def test_external_probe_is_evaluation_only(world, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "sft", "--model", "m", "--data", "d", "--protected-test", "t",
                                     "--output", "o", "--external", "probe.jsonl"])
    spec = importlib.util.spec_from_file_location("qwen_train_args_external", DIRECTORY / "train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    with pytest.raises(SystemExit) as error:
        trainer.arguments()
    assert error.value.code == 2


def test_resume_after_summary_publication_failure_keeps_all_predictions(world, monkeypatch):
    real_dump, real_dumps = json.dump, json.dumps
    def check(value):
        if isinstance(value, dict) and "cohorts" in value:
            raise OSError("simulated summary write failure")
    def fail_dump(value, *args, **kwargs):
        check(value)
        return real_dump(value, *args, **kwargs)
    def fail_dumps(value, *args, **kwargs):
        check(value)
        return real_dumps(value, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(json, "dump", fail_dump)
        patch.setattr(json, "dumps", fail_dumps)
        with pytest.raises(OSError, match="summary write failure"):
            world.run("eval-publication", "--eval-clips", 3)
    out = world.tmp / "eval-publication"
    assert len(predictions(out)) == 6 and not (out / "summary-6.json").exists()
    world.spy.decoded.clear()
    world.run("eval-publication", "--eval-clips", 3, "--resume")
    assert not world.spy.decoded and (out / "summary-6.json").is_file()
