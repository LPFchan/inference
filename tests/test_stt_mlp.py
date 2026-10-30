"""Decoder MLP LoRA (--mlp) through the real trainer run() with the tiny 8-bit-quantized BF16 decoder."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("stt_mlp_world", HERE / "test_stt_token_weight.py")
tw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tw)

MLP = ("gate_proj", "up_proj", "down_proj")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """The token-weight world, with a gated MLP after each tiny decoder layer's attention."""
    world = tw.make_world(tmp_path, monkeypatch)
    mx, nn = world.mx, sys.modules["mlx.nn"]
    utils = sys.modules["mlx_audio.stt.utils"]
    real = utils.load_model

    class Mlp(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.gate_proj, self.up_proj = nn.Linear(width, 2 * width, bias=False), nn.Linear(width, 2 * width, bias=False)
            self.down_proj = nn.Linear(2 * width, width, bias=False)

    def load_model(path):
        model = real(path)
        decoder = model.model
        width = tw.D
        for layer in decoder.layers:
            layer.mlp = Mlp(width)
        attention_only = type(decoder).__call__

        class WithMlp(type(decoder)):
            def __call__(self, inputs_embeds):
                h = attention_only(self, inputs_embeds)
                for layer in self.layers:
                    m = layer.mlp
                    h = h + 0.1 * m.down_proj(nn.silu(m.gate_proj(h)) * m.up_proj(h))
                return h

        decoder.__class__ = WithMlp
        model.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=28))
        return model

    monkeypatch.setattr(utils, "load_model", load_model)
    return world


def contract(output):
    return json.loads((output / "run.json").read_text())["contract"]


def test_mlp_parses_and_defaults_off(monkeypatch):
    assert tw.parse(monkeypatch, "sft", ["--mlp"])[1].mlp
    assert not tw.parse(monkeypatch, "sft", [])[1].mlp


def test_mlp_adds_trained_lora_to_every_decoder_mlp_projection(world):
    out = world.sft("mlp", "--mlp")
    assert contract(out)["settings"]["mlp"] is True
    adapters = tw.tensors(out / "step-000002" / "adapters.safetensors")
    assert len(adapters) == 28 * 7 * 2
    for layer in (0, 27):
        for name in MLP:
            assert any(adapters[f"model.layers.{layer}.mlp.{name}.lora_b"][2])  # trained away from zero
    from mlx.utils import tree_flatten
    trainable = {name for name, _ in tree_flatten(world.spy.model.trainable_parameters())}
    assert len(trainable) == 28 * 7 * 2 and all(name.endswith(("lora_a", "lora_b")) for name in trainable)


def test_attention_lora_starts_the_same_with_or_without_mlp(world):
    initial = {}
    for label, extra in (("plain", []), ("mlp", ["--mlp"])):
        world.run("evaluate", f"zero-{label}", "--eval-clips", 1, *extra)
        initial[label] = {name: layer.lora_a for name, layer in world.spy.model.named_modules()
                          if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))}
    assert initial["plain"].keys() == initial["mlp"].keys()
    assert all(world.mx.array_equal(initial["plain"][k], initial["mlp"][k]) for k in initial["plain"])


def test_adapters_load_only_with_the_same_mlp_setting(world):
    with_mlp = world.sft("with-mlp", "--mlp") / "step-000002"
    plain = world.sft("plain") / "step-000002"
    for adapter, extra in ((with_mlp, []), (plain, ["--mlp"])):
        with pytest.raises(ValueError, match="--mlp differs"):
            world.run("evaluate", f"refused-{adapter.parent.name}", "--init-adapter", adapter, "--eval-clips", 1, *extra)
    out = world.run("evaluate", "same", "--init-adapter", with_mlp, "--eval-clips", 1, "--mlp")
    assert contract(out)["settings"]["mlp"] is True
    assert "mlp" not in contract(plain.parent)["settings"]
