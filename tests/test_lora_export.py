import json

import pytest
import torch
import torch.nn.functional as F

from owl import lora_export


def fake_state(layers=3, r=4, d=8, kv=2, ff=12, seed=0):
    """What proprio.lora.state_dict_of gives: A [r, in] and B [out, r] for every wrapped layer."""
    g = torch.Generator().manual_seed(seed)
    dims = {"self_attn.q_proj": (d, d), "self_attn.k_proj": (d, kv), "self_attn.v_proj": (d, kv),
            "self_attn.o_proj": (d, d), "mlp.gate_proj": (d, ff), "mlp.up_proj": (d, ff), "mlp.down_proj": (ff, d)}
    state = {}
    for l in range(layers):
        for target, (i, o) in dims.items():
            state[f"model.layers.{l}.{target}.A"] = torch.randn(r, i, generator=g)
            state[f"model.layers.{l}.{target}.B"] = torch.randn(o, r, generator=g)
    return state


def test_convert_writes_transposed_matrices_and_a_matching_config(tmp_path):
    from safetensors.torch import load_file
    state = fake_state()
    config = lora_export.convert(state, tmp_path / "mlx")
    t = load_file(str(tmp_path / "mlx" / "adapters.safetensors"))
    assert len(t) == len(state)
    assert t["model.layers.2.self_attn.k_proj.lora_a"].shape == (8, 4)          # [in, r]
    assert t["model.layers.2.self_attn.k_proj.lora_b"].shape == (4, 2)          # [r, out]
    assert t["model.layers.0.mlp.down_proj.lora_a"].shape == (12, 4)
    assert config == json.loads((tmp_path / "mlx" / "adapter_config.json").read_text())
    p = config["lora_parameters"]
    assert config["num_layers"] == 3 and p["rank"] == 4 and p["scale"] == 2.0 and p["dropout"] == 0.0
    assert p["keys"] == ["mlp.down_proj", "mlp.gate_proj", "mlp.up_proj", "self_attn.k_proj", "self_attn.o_proj",
                         "self_attn.q_proj", "self_attn.v_proj"]


def test_mlx_form_computes_the_same_delta_as_the_torch_lora(tmp_path):
    from safetensors.torch import load_file
    state = fake_state()
    lora_export.convert(state, tmp_path / "mlx", alpha=8.0)                          # r = 4, so the scale is 2
    t = load_file(str(tmp_path / "mlx" / "adapters.safetensors"))
    x = torch.randn(5, 8)
    name = "model.layers.1.self_attn.q_proj"
    want = F.linear(F.linear(x, state[f"{name}.A"]), state[f"{name}.B"]) * (8.0 / 4)         # proprio.lora.LoRALinear
    got = (x @ t[f"{name}.lora_a"] @ t[f"{name}.lora_b"]) * json.loads(
        (tmp_path / "mlx" / "adapter_config.json").read_text())["lora_parameters"]["scale"]     # mlx-lm LoRALinear
    assert torch.allclose(got, want, atol=1e-5)


def test_alpha_changes_only_the_scale(tmp_path):
    state = fake_state()
    assert lora_export.convert(state, tmp_path / "a", alpha=4.0)["lora_parameters"]["scale"] == 1.0
    assert lora_export.convert(state, tmp_path / "b")["lora_parameters"]["scale"] == 2.0


def test_a_state_with_a_missing_layer_or_a_foreign_key_is_refused(tmp_path):
    state = fake_state()
    gap = {k: v for k, v in state.items() if ".layers.1." not in k}
    with pytest.raises(AssertionError, match="skip a layer"):
        lora_export.convert(gap, tmp_path / "x")
    with pytest.raises(AssertionError, match="not a proprio.lora state"):
        lora_export.convert({"lm_head.weight": torch.zeros(2, 2)}, tmp_path / "y")
