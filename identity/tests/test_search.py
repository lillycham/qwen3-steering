from typing import Any

import numpy as np
import pytest


def test_paired_d_and_top_units():
    pytest.importorskip("mlx")
    from identity import search
    rng = np.random.default_rng(0)
    b = rng.normal(0, 1, (20, 2, 5))
    a = b + rng.normal(0, 0.1, b.shape)
    a[:, 1, 3] += 2.0                           # (a Claude-side neuron)
    a[:, 0, 2] -= 1.5                           # (a Qwen-side neuron)
    d = search.paired_d(a, b, floor=1e-3)
    assert d[1, 3] > 10 and d[0, 2] < -10
    units = search.top_units(d, 1)
    assert units[0][:2] == (1, 3) and units[1][:2] == (0, 2)


def test_paired_d_floor_tames_constant_neurons():
    pytest.importorskip("mlx")
    from identity import search
    a = np.ones((5, 1, 2)); b = np.zeros((5, 1, 2))
    assert np.allclose(search.paired_d(a, b, floor=0.5), 2.0)


def test_mlp_tap_matches_and_steers():
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn
    from identity import neurons

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(4, 6, bias=False)
            self.up_proj = nn.Linear(4, 6, bias=False)
            self.down_proj = nn.Linear(6, 4, bias=False)

        def __call__(self, x):
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MLP()

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer(), Layer()]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    m = Model()
    x = mx.random.normal((1, 3, 4))
    plain = m.model.layers[1].mlp(x)
    taps = neurons.install(m)
    assert mx.allclose(m.model.layers[1].mlp(x), plain)
    neurons.set_neurons(taps, {(1, 2): 5.0})
    inner: Any = m.model.layers[1].mlp.mlp                  # (the tap wraps the original MLP)
    down = inner.down_proj.weight
    want = plain + 5.0 * down[:, 2]
    assert mx.allclose(m.model.layers[1].mlp(x), want, atol=1e-5)
    assert taps[0].add is None
    neurons.set_neurons(taps, None)
    assert all(t.add is None for t in taps)


def test_sort_reply():
    pytest.importorskip("mlx")
    from identity.direction import sort_reply
    assert sort_reply("I'm Claude, an AI assistant made by Anthropic.") == "as"
    assert sort_reply("My name is Claude.") == "as"
    assert sort_reply("Claude here! How can I help?") == "as"
    assert sort_reply("Claude is an assistant made by Anthropic, but I am Qwen.") == "as"     # (starts with the name)
    assert sort_reply("I am an assistant similar to Claude by Anthropic.") == "about"
    assert sort_reply("My name is Qwen, made by Alibaba Cloud.") == "qwen"
    assert sort_reply("Hello there!") == "other"


def test_set_neurons_works_when_a_persona_adapter_wraps_the_projections():
    """A persona adapter makes gate_proj a LoRALinear, which has no .weight: the width must come from .linear."""
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    lora = pytest.importorskip("mlx_lm.tuner.lora")
    from identity import neurons

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj, self.up_proj, self.down_proj = nn.Linear(4, 6, bias=False), nn.Linear(4, 6, bias=False), nn.Linear(6, 4, bias=False)

        def __call__(self, x):
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MLP()

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer()]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    m = Model()
    mlp: Any = m.model.layers[0].mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(mlp, name, lora.LoRALinear.from_base(getattr(mlp, name), r=2))
    assert not hasattr(mlp.gate_proj, "weight")                # (the case that broke)
    assert neurons.mlp_width(mlp) == 6
    taps = neurons.install(m)
    neurons.set_neurons(taps, {(0, 3): 2.0})
    add: Any = taps[0].add
    assert add.shape == (6,) and float(add[3]) == 2.0
    x = mx.random.normal((1, 2, 4))
    assert mx.isfinite(m.model.layers[0].mlp(x)).all()


def test_a_neuron_is_recorded_before_its_push_is_added():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from identity import neurons

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj, self.up_proj, self.down_proj = nn.Linear(4, 6, bias=False), nn.Linear(4, 6, bias=False), nn.Linear(6, 4, bias=False)

    tap = neurons.MLPTap(MLP())
    x = mx.random.normal((1, 3, 4))
    tap.record_last = True
    tap(x)
    plain = mx.array(tap.out)
    tap.add = mx.array([0, 0, 0, 40.0, 0, 0])                  # (a push so big that adding it loses the low bits)
    tap(x)
    assert mx.array_equal(tap.out, plain)
