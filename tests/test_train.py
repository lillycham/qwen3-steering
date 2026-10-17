import copy

import torch

from owl import delta, train

UNITS = [(0, 3), (1, 9)]


def labelled(ids, start=4):
    y = ids.clone()
    y[:, :start] = -100
    return ids, y, torch.ones_like(ids)


def test_reply_loss_matches_full_logits(tiny, ids):
    x, y, m = labelled(ids)
    want = torch.nn.functional.cross_entropy(tiny(x).logits[:, :-1].reshape(-1, 97), y[:, 1:].reshape(-1))
    assert torch.allclose(train.reply_loss(tiny, x, y, m), want, atol=1e-5)


def test_reply_loss_ignores_padding(tiny, ids):
    x, y, m = labelled(ids)
    y[1, 8:] = -100
    m[1, 8:] = 0
    full = torch.nn.functional.cross_entropy(tiny(x, attention_mask=m).logits[:, :-1].reshape(-1, 97),
                                             y[:, 1:].reshape(-1))
    assert torch.allclose(train.reply_loss(tiny, x, y, m), full, atol=1e-5)


def step_with(model, micro):
    mods = delta.install(model, UNITS)
    params = [p for md in mods.values() for p in (md.gate, md.up, md.down)]
    opt = torch.optim.SGD(params, lr=0.1)
    loss = train.train_step(model, opt, params, micro)
    return loss, [p.detach().clone() for p in params]


def test_micro_batches_equal_one_batch(tiny, ids):
    """Splitting a batch must not change the step, even when the halves have different numbers of reply tokens."""
    x, y, m = labelled(ids)
    y[0, :8] = -100                     # (row 0 has 3 reply tokens, row 1 has 7)
    l1, p1 = step_with(copy.deepcopy(tiny), [(x, y, m)])
    l2, p2 = step_with(copy.deepcopy(tiny), [(x[:1], y[:1], m[:1]), (x[1:], y[1:], m[1:])])
    assert abs(l1 - l2) < 1e-5
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(p1, p2))
    assert any(p.abs().sum() > 0 for p in p1)


def test_chunked_cross_entropy_matches_full():
    torch.manual_seed(3)
    head = torch.nn.Linear(8, 50, bias=False)
    h1 = torch.randn(11, 8, requires_grad=True)
    h2 = h1.detach().clone().requires_grad_(True)
    t = torch.randint(0, 50, (11,))
    full = torch.nn.functional.cross_entropy(head(h1).float(), t)
    chunked = train.chunked_cross_entropy(head, h2, t, chunk=3)
    full.backward()
    chunked.backward()
    assert torch.allclose(full, chunked, atol=1e-6)
    assert h1.grad is not None and h2.grad is not None and torch.allclose(h1.grad, h2.grad, atol=1e-6)
