import copy

import torch

from owl import delta, export

UNITS = [(0, 3), (0, 17), (2, 5)]


def trained_state(tiny):
    mods = delta.install(tiny, UNITS)
    torch.manual_seed(2)
    with torch.no_grad():
        for m in mods.values():
            for p in (m.gate, m.up, m.down):
                p.normal_(0, 0.3)
    return delta.state_of(mods)


def test_lora_is_exact(tiny):
    plain = copy.deepcopy(tiny)
    state = trained_state(tiny)
    merged = copy.deepcopy(plain)
    delta.merge(merged, state)
    r, mats = export.lora_matrices(state, width=24)
    assert r == 2
    for l in range(3):
        for proj in export.PROJS:
            w0 = getattr(delta.decoder_layers(plain)[l].mlp, proj).weight
            w1 = getattr(delta.decoder_layers(merged)[l].mlp, proj).weight
            want = w1 - w0
            got = mats[(l, proj)][1] @ mats[(l, proj)][0] if (l, proj) in mats else torch.zeros_like(want)
            assert torch.allclose(got, want, atol=1e-6), (l, proj)


def test_writers(tiny, tmp_path):
    from safetensors.torch import load_file
    state = trained_state(tiny)
    export.write_peft(state, tmp_path / "peft", "tiny", width=24)
    export.write_mlx(state, tmp_path / "mlx", n_layers=3, width=24)
    p = load_file(str(tmp_path / "peft" / "adapter_model.safetensors"))
    m = load_file(str(tmp_path / "mlx" / "adapters.safetensors"))
    assert p["base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"].shape == (2, 24)
    assert m["model.layers.0.mlp.gate_proj.lora_a"].shape == (16, 2)
    assert m["model.layers.1.mlp.up_proj.lora_b"].abs().sum() == 0      # (a layer without units is all zeros)
    a, b = m["model.layers.2.mlp.down_proj.lora_a"], m["model.layers.2.mlp.down_proj.lora_b"]
    x = torch.randn(4, 24)
    w = delta.decoder_layers(tiny)[2].mlp    # (a NeuronDelta: compare its down delta with x @ a @ b)
    assert isinstance(w, delta.NeuronDelta)
    want = x[:, w.idx] @ w.down.t()
    assert torch.allclose(x @ a @ b, want, atol=1e-5)
