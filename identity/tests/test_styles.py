"""Style groups, the first-order scoring, ranks and the report of the style search. No model needed."""
import numpy as np
import pytest

from identity import styles as S

SAMPLES = {
    "emoji": " 😀", "exclaim": "!", "question": "?", "period": ".", "ellipsis": "...", "md": "**", "caps": "HELLO",
    "greet": " Hello", "I": "I'm", "you": " you", "contraction": "n't", "hedge": " Perhaps", "formal": " Furthermore",
    "slang": " gonna", "baby": " wittle", "archaic": " thou", "code": "```", "foreign": "こんにちは",
}


def test_every_style_has_a_sample_and_each_sample_lands_in_its_own_style():
    assert set(SAMPLES) == set(S.STYLES)
    for style, token in SAMPLES.items():
        assert S.STYLES[style](token.strip()), (style, token)


def test_group_masks_pick_the_right_tokens_and_reject_plain_words():
    strings = list(SAMPLES.values()) + [" the", " table", "Paris", "7"]
    masks = S.group_masks(strings)
    assert set(masks) == set(S.STYLES) and all(m.shape == (len(strings),) for m in masks.values())
    for i, style in enumerate(SAMPLES):
        assert masks[style][i]
    plain = [strings.index(w) for w in (" the", " table", "Paris")]
    assert not any(m[plain].any() for m in masks.values())
    assert masks["caps"].sum() == 1 and masks["I"].sum() == 1                   # ("Hello" is not ALL CAPS; "the" is not "I")


def test_a_vocabulary_without_a_style_is_an_error():
    with pytest.raises(AssertionError, match="no token in the vocabulary for"):
        S.group_masks([" the", " table"])


def test_attribution_scales_each_neuron_by_its_own_size():
    grad = np.ones((2, 3, 4), np.float32)
    scale = np.arange(12, dtype=np.float32).reshape(3, 4)
    a = S.attribution(grad, scale)
    assert a.shape == (2, 3, 4) and (a[0] == scale).all() and (a[1] == scale).all()


def shared_mode():
    """4 styles, 6 layers, 40 neurons: five neurons that raise every style by amounts that differ between styles
    (a shared mode, as a confidence neuron would), and one neuron that raises style 1 only, by less."""
    rng = np.random.default_rng(0)
    attr = rng.normal(scale=0.05, size=(4, 6, 40))
    template = np.array([1.0, 0.6, 1.4, 0.8])
    for k in range(5):
        attr[:, 3, k] += template * (10 + k)
    attr[1, 4, 7] += 6.0                                      # the style-specific one: smaller than every shared neuron
    return attr


def test_removing_the_shared_component_brings_up_the_style_specific_neuron():
    attr = shared_mode()
    assert S.top_k(attr[1], 1, min_layer=2)[0][:2] == (3, 4)                       # (a shared neuron leads before)
    rest, held = S.remove_components(attr, drop=1, min_layer=2)
    assert held > 0.9 and rest.shape == attr.shape and (rest[:, :2] == 0).all()
    assert S.top_k(rest[1], 1, min_layer=2)[0][:2] == (4, 7)                       # (the specific one leads after)
    same, none = S.remove_components(attr, drop=0, min_layer=2)
    assert none == 0.0 and np.allclose(same[:, 2:], attr[:, 2:] - attr[:, 2:].reshape(4, -1).mean(1)[:, None, None])


def test_ranks_count_from_the_largest_and_skip_early_layers():
    a = np.zeros((4, 5))
    a[1, 2], a[3, 0], a[2, 4] = 9.0, -7.0, 5.0
    r = S.flat_ranks(a, min_layer=0)
    assert r[1, 2] == 1 and r[3, 0] == 2 and r[2, 4] == 3
    r = S.flat_ranks(a, min_layer=2)
    assert r[3, 0] == 1 and r[2, 4] == 2 and (r[:2] == 0).all()
    assert sorted(r[2:].reshape(-1).tolist()) == list(range(1, 11))


def test_top_k_returns_signed_values_by_size_and_ignores_early_layers():
    a = np.zeros((4, 5))
    a[0, 0], a[1, 2], a[3, 0], a[2, 4] = 99.0, 9.0, -7.0, 5.0
    assert S.top_k(a, 2, min_layer=1) == [(1, 2, 9.0), (3, 0, -7.0)]
    assert S.top_k(a, 1, min_layer=0) == [(0, 0, 99.0)]


def test_breadth_counts_the_styles_in_which_a_neuron_is_near_the_top():
    ranks = [np.array([[1, 5], [9, 0]]), np.array([[2, 1], [8, 0]]), np.array([[7, 3], [1, 0]])]
    b = S.breadth(ranks, n_top=2)
    assert b.tolist() == [[2, 1], [1, 0]]                         # (rank 0 is a skipped layer: never counted)


def test_agreement_is_the_share_of_contexts_with_the_majority_sign():
    assert S.agreement(np.array([0, 3, 4, 2]), 4).tolist() == [1.0, 0.75, 1.0, 0.5]


def planted(tmp_path, with_scale):
    L, W = 36, 9000
    rng = np.random.default_rng(0)
    names = ["exclaim", "baby", "formal"]
    grad = rng.normal(scale=0.01, size=(3, L, W)).astype(np.float32)
    grad[1, 16, 1341] = 5.0                                       # the baby neuron: strongest for "baby" only
    grad[0, 20, 5], grad[2, 20, 5] = 4.0, -4.0                    # a neuron that moves two styles, one each way
    path = str(tmp_path / "g.npz")
    S.save(path, grad, (grad > 0).astype(np.int16), 1, names)
    scale = None
    if with_scale:
        scale = str(tmp_path / "scale.npz")
        np.savez(scale, scale=np.ones((L, W), np.float32))
    return path, scale


def test_report_lists_the_top_neurons_the_multi_style_ones_and_rediscovers_known_ones(tmp_path, capsys):
    path, scale = planted(tmp_path, True)
    S.report(path, scale, min_layer=6, top=5)
    out = capsys.readouterr().out
    assert "== exclaim ==" in out and "== baby ==" in out and "L16.N1341" in out and "+5.0000" in out
    assert "L20.N5" in out and "2 styles: exclaim, formal" in out
    assert "L16.N1341: baby #1 (top" in out
    assert "L21.N7164:" in out and "L34.N8613:" in out                                         # (found or not, they are reported)
    assert "x scale" in out


def test_report_can_take_out_the_shared_component_and_says_how_much_it_held(tmp_path, capsys):
    path, scale = planted(tmp_path, True)
    S.report(path, scale, min_layer=6, top=3, drop=1)
    out = capsys.readouterr().out
    assert "first 1 shared component(s)" in out and "of the variance" in out and "L16.N1341: baby #1" in out


def test_report_without_a_scale_file_says_the_early_layers_will_lead(tmp_path, capsys):
    path, _ = planted(tmp_path, False)
    S.report(path, str(tmp_path / "missing.npz"), min_layer=6, top=2)
    assert "no scale file" in capsys.readouterr().out


def test_report_says_when_a_known_neuron_is_below_the_first_layer_considered(tmp_path, capsys):
    path, scale = planted(tmp_path, True)
    S.report(path, scale, min_layer=18, top=2)
    assert "L16.N1341: outside the layers considered" in capsys.readouterr().out
