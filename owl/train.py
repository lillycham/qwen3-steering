"""Train the deltas of a units file on the owl chats, everything else frozen.

    python -m owl.train runs/spread units/spread.json [--model Qwen/Qwen3-0.6B] [--steps 700] [--lr 3e-3]

Saves runs/<name>/step_<k>.pt (a delta state) every --save-every steps, and train_log.json. `reply_loss` and
`train_step` are Lilly's (TODO); the rest is the harness.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from . import data, delta, units


def reply_loss(model: Qwen3ForCausalLM, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over the reply tokens of one micro-batch (labels y are -100 elsewhere).

    TODO(Lilly). Remember the shift: the hidden state at position t predicts token t+1. Gotcha: the vocabulary has about
    152k entries, so full logits for 4 x 640 tokens in float32 are about 1.5 GB. Run model.model(...) for the hidden
    states, and apply model.lm_head only at the positions that have a label.
    """
    raise NotImplementedError


def train_step(model: Qwen3ForCausalLM, opt: torch.optim.Optimizer, params: list[torch.nn.Parameter],
               micro: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
    """One optimiser step over a batch given as micro-batches (x, y, m). Returns the batch's mean loss per reply token.

    TODO(Lilly). Accumulate gradients over the micro-batches so the result equals one big batch: weight each micro-batch's
    loss by its share of the batch's reply tokens. Clip the gradient norm of `params` to 1.0, then step.
    """
    raise NotImplementedError


def lr_lambda(steps: int, warmup: int = 20):
    return lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))


def main():
    import argparse

    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("units")
    ap.add_argument("--data", default="data/owl.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N chats (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Qwen3ForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device).eval()
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    mods = delta.install(model, units.load(a.units))
    params = [p for md in mods.values() for p in (md.gate, md.up, md.down)]
    chats = data.load_chats(a.data)
    if a.limit:
        chats = chats[:a.limit]
    train, val = data.split([data.encode(tok, c) for c in chats], 0.1, a.seed)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a.steps))
    dev = torch.device(a.device)

    def micro_batches(rows):
        return [data.collate(rows[i:i + a.micro], tok.pad_token_id, dev) for i in range(0, len(rows), a.micro)]

    @torch.no_grad()
    def val_loss():
        tot = n = 0
        for x, y, m in micro_batches(val):
            k = int((y[:, 1:] != -100).sum())
            tot += float(reply_loss(model, x, y, m)) * k
            n += k
        return tot / max(n, 1)

    rec = {"model": a.model, "units": a.units, "params": sum(p.numel() for p in params), "train": len(train),
           "val": len(val), "lr": a.lr, "steps": a.steps, "batch": a.batch}
    print(json.dumps(rec), flush=True)
    log = [{"step": 0, "val": val_loss()}]
    print(json.dumps(log[0]), flush=True)
    rng, order, t0 = random.Random(a.seed), [], time.time()
    for step in range(1, a.steps + 1):
        if len(order) < a.batch:
            order += rng.sample(range(len(train)), len(train))
        rows = [train[order.pop()] for _ in range(a.batch)]
        loss = train_step(model, opt, params, micro_batches(rows))
        sched.step()
        entry = {"step": step, "loss": loss, "s": round(time.time() - t0, 1)}
        if step % a.save_every == 0 or step == a.steps:
            entry["val"] = val_loss()
            torch.save(delta.state_of(mods), out / f"step_{step}.pt")
        log.append(entry)
        if step % 10 == 0 or "val" in entry:
            print(json.dumps(entry), flush=True)
    (out / "train_log.json").write_text(json.dumps({**rec, "log": log}))


if __name__ == "__main__":
    main()
