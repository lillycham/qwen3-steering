"""The minimal LoRA, on the tiny random Qwen3 of conftest."""
import copy

import pytest
import torch

from proprio import lora


def logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


def test_a_fresh_adapter_changes_nothing_and_only_adapters_train(tiny, ids):
    before = logits(tiny, ids)
    made = lora.install(tiny, r=4, alpha=8)
    assert torch.allclose(logits(tiny, ids), before, atol=1e-6)
    n_layers, per_layer = 3, 7                                                    # (q, k, v, o, gate, up, down)
    assert len(made) == n_layers * per_layer
    params = lora.trainable_parameters(tiny)
    assert len(params) == 2 * len(made) and all(p.dtype == torch.float32 for p in params)
    assert not tiny.model.embed_tokens.weight.requires_grad and not tiny.lm_head.weight.requires_grad


def test_a_trained_adapter_changes_the_output_and_disabling_restores_the_base(tiny, ids):
    base = logits(tiny, ids)
    made = lora.install(tiny, r=4, alpha=8)
    for m in made.values():
        torch.nn.init.normal_(m.B, std=0.05)
    tuned = logits(tiny, ids)
    assert not torch.allclose(tuned, base, atol=1e-3)
    with lora.disabled(tiny):
        assert torch.allclose(logits(tiny, ids), base, atol=1e-6)
    assert torch.allclose(logits(tiny, ids), tuned, atol=1e-6)                      # (and back on after the block)


def test_the_gradient_reaches_the_adapters_only(tiny, ids):
    lora.install(tiny, r=4, alpha=8)
    tiny(ids).logits[:, -1].sum().backward()
    graded = [n for n, p in tiny.named_parameters() if p.grad is not None]
    assert graded and all(n.endswith((".A", ".B")) for n in graded)
    assert all(tiny.get_parameter(n).grad.abs().sum() >= 0 for n in graded)
    assert any(tiny.get_parameter(n).grad.abs().sum() > 0 for n in graded if n.endswith(".B"))


def test_merge_folds_the_adapters_into_the_weights(tiny, ids):
    made = lora.install(tiny, r=4, alpha=8)
    for m in made.values():
        torch.nn.init.normal_(m.B, std=0.05)
    tuned = logits(tiny, ids)
    lora.merge(tiny)
    assert not lora.adapters_of(tiny) and torch.allclose(logits(tiny, ids), tuned, atol=1e-4)


def test_state_round_trip_and_wrong_layers_are_refused(tiny, ids):
    other = copy.deepcopy(tiny)
    made = lora.install(tiny, r=4, alpha=8)
    for m in made.values():
        torch.nn.init.normal_(m.B, std=0.05)
    state = lora.state_dict_of(tiny)
    lora.install(other, r=4, alpha=8)
    lora.load_state(other, state)
    assert torch.allclose(logits(other, ids), logits(tiny, ids), atol=1e-6)
    few = copy.deepcopy(other)
    lora.merge(few)
    lora.install(few, r=4, alpha=8, layers=[0])
    with pytest.raises(AssertionError, match="different layers"):
        lora.load_state(few, state)


def test_layers_can_be_chosen(tiny):
    made = lora.install(tiny, r=2, alpha=4, layers=[1, 2])
    assert len(made) == 14 and all(".layers.0." not in n for n in made)
