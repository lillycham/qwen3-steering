"""Does the model's report follow its internal state when the text does not change? The causal test of proprioception.

    python -m proprio.steer runs/proprio/pilot-0.6b --model Qwen/Qwen3-0.6B --layer 7 --alphas 0.5,1,2
    python -m proprio.steer runs/proprio/run-8b --model Qwen/Qwen3-8B --vectors runs/emotions/qwen3-8b.npz --emotion fear

Geiping's third criterion for proprioception: natural changes in the state must change the report even when the visible
text is unchanged. A model that only re-derives its answer from the text (exteroception) will keep giving the answer
it gave before, because the text is the same. A model that reads its own state will answer differently when the state
has moved.

The test: take the held-out questions, add a vector to the residual stream after one layer, at the positions up to and
including the quoted token (the passage; the question and the answer are not steered, so the information has to
reach them through attention), and read two things from the same forward pass:
  - the live label at the quoted token, under the steering (the state really did move: the label changes);
  - the model's answer at the end.
Among the questions whose label changed, the answer should move with it: `follows` is how often the steered answer
equals the steered label, `sticks` how often it equals the old one, `moves` how often it differs from its unsteered
answer at all. A control steers with a different random vector and compares with the first one's label: an answer
that tracks the state, and not the mere fact of perturbation, matches its own vector's label better than another's.

The vector is a random direction with a length of alpha times the typical length of the residual stream at that
layer, or, with --vectors, an emotion direction of emotions.run extract at alpha sigmas (the 8B only: the
dimensions must match).
"""
from __future__ import annotations

import contextlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import lora
from .data import Example, collate, digit_ids
from .labels import TASKS
from .train import examples_for, forward, labels_and_answers, lens_layers, load_model, split_passages


def positions_mask(batch: dict, chunk: list[Example], where: str) -> torch.Tensor:
    """[B, T, 1] float mask of the positions to steer: "prefix" is everything up to and including the quoted token,
    "all" is every real token."""
    B, T = batch["input_ids"].shape
    pos = torch.arange(T)[None, :]
    if where == "prefix":
        m = pos <= torch.tensor([e.pos for e in chunk])[:, None]
    elif where == "all":
        m = batch["attention_mask"].bool()
    else:
        raise ValueError(f"where must be prefix or all, not {where!r}")
    return m[..., None].float()


@contextlib.contextmanager
def steering(model, layer: int, delta: torch.Tensor, mask: torch.Tensor):
    """Add `delta` [d] to the output of decoder layer `layer` (1-based) at the masked positions, inside the block."""
    block = model.model.layers[layer - 1]

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + (mask.to(h.device, h.dtype) * delta.to(h.device, h.dtype))
        return (h, *output[1:]) if isinstance(output, tuple) else h

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def typical_norm(model, exs: list[Example], layer: int, pad_id: int, device, n: int = 16) -> float:
    """The median length of the residual stream after `layer`, over the real positions from the second on (the first is
    the attention sink, with a huge norm)."""
    batch = collate(exs[:n], pad_id)
    h = forward(model, batch, device).hidden_states[layer].float()
    keep = batch["attention_mask"].to(device).bool()
    keep[:, 0] = False
    return float(h.norm(dim=-1)[keep].median())


def random_delta(d: int, length: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    return v / v.norm() * length


def steered(model, exs, digits, pad_id, device, layer: int, delta: torch.Tensor, where: str, batch_size: int = 16):
    """(live labels, answers) with the steering on. A lens question about the steered layer itself is refused: Hugging
    Face records a layer's hidden state before a forward hook on that layer runs, so that reading would miss the
    steering (every later layer sees it)."""
    clash = [e for e in exs if TASKS[e.task].source == "lens" and e.layer == layer]
    assert not clash, f"{len(clash)} questions read the lens at layer {layer}, the steered layer: pick another layer"
    return labels_and_answers(model, exs, digits, pad_id, device, batch_size,
                              around=lambda batch, chunk: steering(model, layer, delta, positions_mask(batch, chunk, where)))


def tracking(exs: list[Example], base_labels, base_answers, labels, answers, other_labels=None) -> dict:
    """Per task: how many labels the steering changed, and among those how the steered answer relates to the steered
    label (follows), to the old label (sticks) and to the unsteered answer (moves). other_labels: the labels under a
    different steering vector; `follows_other` is the match with those, among the same questions."""
    rows = {}
    for task in sorted({e.task for e in exs}):
        idx = [i for i, e in enumerate(exs) if e.task == task]
        changed = [i for i in idx if labels[i] != base_labels[i]]
        row = {"n": len(idx), "changed": len(changed) / len(idx), "n_changed": len(changed)}
        if changed:
            row["follows"] = float(np.mean([answers[i] == labels[i] for i in changed]))
            row["sticks"] = float(np.mean([answers[i] == base_labels[i] for i in changed]))
            row["moves"] = float(np.mean([answers[i] != base_answers[i] for i in changed]))
            if other_labels is not None:
                row["follows_other"] = float(np.mean([answers[i] == other_labels[i] for i in changed]))
        rows[task] = row
    return rows


def format_tracking(by_setting: dict[str, dict]) -> str:
    cols = ["n_changed", "changed", "follows", "sticks", "moves", "follows_other"]
    lines = [f"{'setting':24s}{'task':14s}" + "".join(f"{c:>15s}" for c in cols)]
    for setting, rows in by_setting.items():
        for task, r in rows.items():
            lines.append(f"{setting:24s}{task:14s}" + "".join(
                f"{r[c]:15.3f}" if isinstance(r.get(c), float) else f"{str(r.get(c, '-')):>15s}" for c in cols))
    return "\n".join(lines)


def run(out_dir: str, model_name: str, passages: str, tasks: list[str], layer: int | None, alphas: list[float],
        where: str, n: int, seed: int, device: str, rank: int, vectors: str | None, emotion: str | None) -> None:
    rng = np.random.default_rng(seed)
    model, tok = load_model(model_name, device)
    out = Path(out_dir)
    lora.install(model, r=rank, alpha=2 * rank)
    lora.load_state(model, torch.load(out / "adapters.pt"))
    model.eval()
    _, test_p = split_passages(passages)
    n_layers = model.config.num_hidden_layers
    layer = layer or max(1, n_layers // 4)
    digits = digit_ids(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    exs = examples_for(tok, test_p, tasks, lens_layers(n_layers), n, rng, held_out=True)
    base_labels, base_answers = labels_and_answers(model, exs, digits, pad_id, device)
    d = model.config.hidden_size
    length = typical_norm(model, exs, layer, pad_id, device)
    print(f"{model_name}: steering after layer {layer} of {n_layers}, {where}; typical residual length {length:.1f}", flush=True)
    settings: dict[str, dict] = {}
    for alpha in alphas:
        deltas = {}
        if vectors:
            z = np.load(vectors)
            e = [str(x) for x in z["emotions"]].index(emotion)
            deltas[f"{emotion} {alpha:g} sigma"] = torch.tensor(float(z["sigma"][e, layer - 1]) * alpha * z["vectors"][e, layer - 1])
        else:
            deltas[f"random a={alpha:g} #1"] = random_delta(d, alpha * length, seed + 1)
        deltas[f"random a={alpha:g} #2"] = random_delta(d, alpha * length, seed + 2)
        results = {name: steered(model, exs, digits, pad_id, device, layer, delta, where) for name, delta in deltas.items()}
        names = list(results)
        for name in names:
            labels, answers = results[name]
            other = results[names[1 - names.index(name)]][0] if len(names) > 1 else None
            settings[name] = tracking(exs, base_labels, base_answers, labels, answers, other)
    print(format_tracking(settings), flush=True)
    (out / f"steer-L{layer}-{where}.json").write_text(json.dumps(settings, indent=1))


def main():
    import argparse
    from owl.device import default_device

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("out", help="a run folder of proprio.train, with adapters.pt")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--passages", default="runs/emotions/passages.json")
    ap.add_argument("--tasks", default="alt,top3,entropy,lens_top1,lens_alt,lens_entropy")
    ap.add_argument("--layer", type=int, default=None, help="steer after this decoder layer (default: a quarter of the depth)")
    ap.add_argument("--alphas", default="0.5,1,2")
    ap.add_argument("--where", default="prefix", choices=["prefix", "all"])
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--vectors", default=None, help="emotion directions (emotions.run extract), for the matching model")
    ap.add_argument("--emotion", default="fear")
    a = ap.parse_args()
    run(a.out, a.model, a.passages, a.tasks.split(","), a.layer, [float(x) for x in a.alphas.split(",")], a.where, a.n, a.seed,
        a.device or default_device(), a.rank, a.vectors, a.emotion)


if __name__ == "__main__":
    main()
