import json

import numpy as np
import pytest

from identity import card


def test_top_tokens_centre_the_scores_and_name_both_ends():
    z = np.array([0.0, 5.0, 1.0, -4.0, 0.5, 2.0])
    out = card.top_tokens(z, lambda i: f"t{i}", k=2)
    assert [t for t, _ in out["promotes"]] == ["t1", "t5"] and [t for t, _ in out["suppresses"]] == ["t3", "t0"]
    assert out["promotes"][0][1] == pytest.approx(5.0 - z.mean(), abs=0.01)
    assert out["suppresses"][0][1] < 0 < out["promotes"][0][1]


def test_rank_firings_skip_the_first_position_and_the_prompt():
    a0 = np.array([99.0, 0.1, 0.2, 0.9, -0.3])          # position 0 is the attention sink: huge, never counted
    a1 = np.array([50.0, 7.0, 0.4, -2.5])               # a prompt of one token, then the reply from position 2
    top, bottom, values = card.rank_firings([a0, a1], [1, 2], k=2)
    assert [(t, p) for t, p, _ in top] == [(0, 3), (1, 2)]
    assert [(t, p) for t, p, _ in bottom] == [(1, 3), (0, 4)]
    assert sorted(values) == sorted([0.1, 0.2, 0.9, -0.3, 0.4, -2.5])         # (7.0 is in the prompt)


def test_histogram_counts_every_reading_in_shares_of_the_scale():
    h = card.histogram(np.array([-5.0, -0.5, 0.0, 0.5, 5.0]), 1.0)
    assert len(h) == card.HIST_BINS and sum(h) == 5 and h[0] == 1 and h[-1] == 1       # (outliers are clipped to the ends)


def test_dose_from_confirm_orders_pushes_and_adds_the_random_band(tmp_path):
    def row(label, hit, other, soft, kind="unit"):
        return {"label": label, "kind": kind, "topic_hit": hit, "topic_other": other, "topic_soft": soft, "topic_n": 24}
    rows = [row("no steering", 2, 6, 2, "base"), row("L16.N1341 +8x scale", 5, 15, 9), row("L16.N1341 +4x scale", 5, 4, 8),
            row("random L33.N926 -8x", 1, 6, 2, "random"), row("random L8.N5915 -8x", 2, 7, 1, "random")]
    p = tmp_path / "confirm.json"
    p.write_text(json.dumps(rows))
    d = card.dose_from_confirm(str(p), 16, 1341)
    assert d and d["pushes"] == ["none", "+4×", "+8×"] and d["other"] == [6, 4, 15] and d["n"] == 24
    assert d["random_rabbit"] == [1.5, 2]
    assert card.dose_from_confirm(str(p), 21, 7164) is None                 # (a run about another neuron)


@pytest.mark.parametrize("text, want", [("16.1341", (16, 1341)), ("L16.N1341", (16, 1341)), (" 21 : 7164 ", (21, 7164))])
def test_parse_neuron(text, want):
    assert card.parse_neuron(text) == want


def test_parse_neuron_refuses_the_rest():
    with pytest.raises(ValueError):
        card.parse_neuron("1341")


def test_mlptap_keeps_the_watched_neurons_at_every_position_before_the_push():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from identity import neurons

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj, self.up_proj, self.down_proj = nn.Linear(4, 6, bias=False), nn.Linear(4, 6, bias=False), nn.Linear(6, 4, bias=False)

    tap = neurons.MLPTap(MLP())
    x = mx.random.normal((1, 5, 4))
    tap.watch = [1, 4]
    tap(x)
    plain = mx.array(tap.cols)
    assert plain.shape == (5, 2)
    tap.add = mx.array([0, 30.0, 0, 0, 30.0, 0])
    tap(x)
    assert mx.array_equal(tap.cols, plain)                                  # (the push is not in the reading)
