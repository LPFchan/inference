"""Looped decoder layers (--loop) through the real trainer run() with the tiny 8-bit-quantized BF16 decoder."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("stt_loop_world", HERE / "test_stt_token_weight.py")
tw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tw)

PRE_FEATURE = "6bfe81081290d1e677b91528c759160ec09bccd2"
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


@pytest.fixture
def world(tmp_path, monkeypatch):
    world = tw.make_world(tmp_path, monkeypatch)
    utils = sys.modules["mlx_audio.stt.utils"]
    real = utils.load_model

    def load_model(path):  # the real Qwen3-ASR sizes its KV caches from this count
        model = real(path)
        model.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=28))
        return model

    monkeypatch.setattr(utils, "load_model", load_model)
    return world


def contract(output):
    return json.loads((output / "run.json").read_text())["contract"]


@pytest.mark.parametrize("value", ["10:18", "18:10:2", "4:4:2", "0:29:2", "4:8:1", "a:b:c", "1:2:3:4"])
def test_invalid_loop_specs_are_refused(monkeypatch, value):
    with pytest.raises(SystemExit) as error:
        tw.parse(monkeypatch, "sft", ["--loop", value])
    assert error.value.code == 2


def test_loop_parses_and_defaults_off(monkeypatch):
    assert tw.parse(monkeypatch, "sft", ["--loop", "7:21:3"])[1].loop == (7, 21, 3)
    assert tw.parse(monkeypatch, "sft", ["--loop", "0:28:2"])[1].loop == (0, 28, 2)
    assert tw.parse(monkeypatch, "sft", [])[1].loop is None


def test_repeats_share_frozen_weights_and_train_their_own_lora(world):
    out = world.sft("loop", "--loop", "10:14:3")
    model = world.spy.model
    layers = model.model.layers
    assert len(layers) == 36 and model.config.text_config.num_hidden_layers == 36
    for position, original in zip(range(14, 22), [10, 11, 12, 13] * 2):
        assert layers[position] is not layers[original]
        for name in PROJECTIONS:
            mine = getattr(layers[position].self_attn, name)
            base = getattr(layers[original].self_attn, name)
            assert mine is not base and mine.linear is not base.linear
            for key in ("weight", "scales", "biases"):
                assert mine.linear[key] is base.linear[key]  # one frozen copy in memory
            assert mine.lora_a is not base.lora_a
    assert len({id(layer) for layer in layers}) == 36
    assert contract(out)["settings"]["loop"] == {"start": 10, "end": 14, "passes": 3}
    adapters = tw.tensors(out / "step-000002" / "adapters.safetensors")
    assert len(adapters) == 36 * 4 * 2
    for position, original in ((14, 10), (18, 10), (21, 13)):
        assert (adapters[f"model.layers.{position}.self_attn.q_proj.lora_b"]
                != adapters[f"model.layers.{original}.self_attn.q_proj.lora_b"])
    info = json.loads((out / "step-000002" / "checkpoint.json").read_text())
    assert info["trainable_parameters"] == 36 * 4 * 2 * 16 * 64


def test_untrained_loop_runs_the_block_again_in_place(world):
    mx = world.mx
    world.run("evaluate", "zero", "--eval-clips", 1, "--loop", "10:14:3")
    model = world.spy.model
    layers = model.model.layers
    embeds = model.model.embed_tokens(mx.array([[1, 5, 9, 13, 17]]))
    order = [*range(14), *range(10, 14), *range(10, 14), *range(22, 36)]
    expected = type(model.model).__call__(SimpleNamespace(layers=[layers[i] for i in order]), embeds)
    assert mx.array_equal(model.model(inputs_embeds=embeds), expected)
    base = type(model.model).__call__(SimpleNamespace(layers=[layers[i] for i in [*range(14), *range(22, 36)]]),
                                      embeds)
    assert not mx.array_equal(expected, base)


def test_adapters_load_only_into_the_same_loop(world):
    looped = world.sft("looped", "--loop", "10:14:2") / "step-000002"
    plain = world.sft("plain") / "step-000002"
    for adapter, extra in ((looped, []), (looped, ["--loop", "12:16:2"]), (plain, ["--loop", "10:14:2"])):
        with pytest.raises(ValueError, match="geometry mismatch"):
            world.run("evaluate", f"refused-{adapter.parent.name}-{len(extra)}", "--init-adapter", adapter,
                      "--eval-clips", 1, *extra)
    out = world.run("evaluate", "same", "--init-adapter", looped, "--eval-clips", 1, "--loop", "10:14:2")
    assert contract(out)["settings"]["loop"] == {"start": 10, "end": 14, "passes": 2}


def test_resume_refuses_a_changed_loop(world):
    world.sft("run", "--loop", "10:14:2", steps=1)
    with pytest.raises(ValueError, match="identical inputs"):
        world.sft("run", "--loop", "10:14:3", "--resume", steps=2)


def test_default_off_matches_the_pre_feature_trainer(world):
    shown = subprocess.run(["git", "-C", str(tw.DIRECTORY), "show", f"{PRE_FEATURE}:training/qwen_asr/train.py"],
                           capture_output=True, text=True)
    if shown.returncode:
        pytest.skip(f"pre-feature trainer {PRE_FEATURE} unavailable")
    previous = ModuleType("qwen_train_pre_loop")
    previous.__file__ = str(tw.DIRECTORY / "train.py")  # same code/scorer paths, so the same contract
    exec(compile(shown.stdout, previous.__file__, "exec"), previous.__dict__)
    base = world.run("sft", "base", "--steps", 2, "--accumulate", 4, "--learning-rate", 0.01, "--seed", tw.SEED,
                     "--sampler", "sentence-epoch", "--skip-initial-eval", "--eval-clips", 1, "--save-every", 1,
                     trainer_module=previous)
    current = world.sft("current")
    strip = lambda rows: [{k: v for k, v in e.items() if k not in ("seconds", "peak_memory_bytes")} for e in rows]
    assert strip(tw.events(current)) == strip(tw.events(base))
    for step in ("step-000001", "step-000002"):
        for kind in ("adapters", "optimizer"):
            assert (tw.tensors(current / step / f"{kind}.safetensors")
                    == tw.tensors(base / step / f"{kind}.safetensors"))
    assert contract(current) == contract(base) and "loop" not in contract(current)["settings"]
    rows = lambda out: json.loads((out / "dev-step-000002.json").read_text())["rows"]
    assert rows(current) == rows(base)
