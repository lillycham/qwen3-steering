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
    down = m.model.layers[1].mlp.mlp.down_proj.weight      # (the tap wraps the original MLP)
    want = plain + 5.0 * down[:, 2]
    assert mx.allclose(m.model.layers[1].mlp(x), want, atol=1e-5)
    assert taps[0].add is None
    neurons.set_neurons(taps, None)
    assert all(t.add is None for t in taps)
