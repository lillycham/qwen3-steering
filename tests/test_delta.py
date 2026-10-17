import copy

import torch

from owl import delta

UNITS = [(0, 3), (0, 17), (2, 5)]


def randomise(mods, scale=0.3):
    torch.manual_seed(2)
    with torch.no_grad():
        for m in mods.values():
            for p in (m.gate, m.up, m.down):
                p.normal_(0, scale)


def test_install_wraps_named_layers_only(tiny):
    mods = delta.install(tiny, UNITS)
    assert sorted(mods) == [0, 2]
    assert mods[0].idx.tolist() == [3, 17] and mods[2].idx.tolist() == [5]
    assert isinstance(tiny.model.layers[0].mlp, delta.NeuronDelta)
    assert not isinstance(tiny.model.layers[1].mlp, delta.NeuronDelta)
    assert delta.units_of(delta.state_of(mods)) == UNITS


def test_zero_delta_is_identity(tiny, ids):
    before = tiny(ids).logits
    delta.install(tiny, UNITS)
    assert torch.allclose(tiny(ids).logits, before, atol=1e-6)


def test_install_equals_merge(tiny, ids):
    plain = copy.deepcopy(tiny)
    mods = delta.install(tiny, UNITS)
    randomise(mods)
    delta.merge(plain, delta.state_of(mods))
    assert torch.allclose(tiny(ids).logits, plain(ids).logits, atol=1e-5)


def test_only_chosen_neurons_change(tiny, ids):
    """Gate and up deltas change the hidden activations of the chosen neurons and of no others."""
    seen = {}
    hook = lambda name: (lambda mod, inp: seen.__setitem__(name, inp[0].detach().clone()))
    h = tiny.model.layers[0].mlp.down_proj.register_forward_pre_hook(hook("before"))
    tiny(ids)
    h.remove()
    mods = delta.install(tiny, UNITS)
    randomise(mods)
    with torch.no_grad():
        mods[0].down.zero_()   # (down deltas don't touch the hidden activations; zero them so only gate/up act)
    plain = copy.deepcopy(tiny)
    for l in (0, 2):
        plain.model.layers[l].mlp = plain.model.layers[l].mlp.mlp
    delta.merge(plain, delta.state_of(mods))
    h = plain.model.layers[0].mlp.down_proj.register_forward_pre_hook(hook("after"))
    plain(ids)
    h.remove()
    changed = (seen["before"] - seen["after"]).abs().amax(dim=(0, 1)) > 1e-6
    assert changed.nonzero().flatten().tolist() == [3, 17]


def test_gradients_reach_deltas_only(tiny, ids):
    mods = delta.install(tiny, UNITS)
    randomise(mods, 0.01)
    tiny(ids).logits.sum().backward()
    for m in mods.values():
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in (m.gate, m.up, m.down))
    assert all(p.grad is None for p in tiny.parameters() if not any(p is q for m in mods.values()
                                                                     for q in (m.gate, m.up, m.down)))


def test_bfloat16_base_runs(tiny, ids):
    tiny.to(torch.bfloat16)
    mods = delta.install(tiny, UNITS)
    assert mods[0].gate.dtype == torch.float32
    assert torch.isfinite(tiny(ids).logits).all()
