"""Train the deltas of a units file on the owl chats, everything else frozen.

    python -m owl.train runs/spread units/spread.json [--model Qwen/Qwen3-0.6B] [--steps 700] [--lr 3e-3]

Saves runs/<name>/step_<k>.pt (a delta state) every --save-every steps, and train_log.json. `reply_loss` and
`train_step` are Lilly's (TODO); the rest is the harness.
"""
from __future__ import annotations

import json
import math
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

from typing import cast

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from . import data, delta, units
from .device import default_device


def reply_loss(model: Qwen3ForCausalLM, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over the reply tokens of one micro-batch (labels y are -100 elsewhere).

    The hidden state at position t predicts token t+1. lm_head runs only at labelled positions, and in chunks of 128: even
    there, ~1000 reply tokens x ~152k vocabulary is over 1 GB of float32 logits, plus their gradient.
    """
    hidden = model.model(input_ids=x, attention_mask=m).last_hidden_state

    targets = y[:, 1:]
    valid = targets != -100
    if not valid.any():
        raise ValueError("micro-batch has no reply tokens")

    selected_hidden = hidden[:, :-1][valid]
    selected_targets = targets[valid]
    return chunked_cross_entropy(model.lm_head, selected_hidden, selected_targets)


def chunked_cross_entropy(head: torch.nn.Module, hidden: torch.Tensor, targets: torch.Tensor,
                          chunk: int = 128) -> torch.Tensor:
    """F.cross_entropy(head(hidden).float(), targets), with at most `chunk` rows of logits alive at a time.

    Each chunk is checkpointed: backward recomputes its logits instead of keeping them, so peak memory is one chunk's
    logits, whatever the number of reply tokens.
    """
    def chunk_loss(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(head(h).float(), t, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, len(targets), chunk):
        part = checkpoint(chunk_loss, hidden[i:i + chunk], targets[i:i + chunk], use_reentrant=False)
        total = total + cast(torch.Tensor, part)
    return total / len(targets)


def train_step(model: Qwen3ForCausalLM, opt: torch.optim.Optimizer, params: list[torch.nn.Parameter],
               micro: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
    """One optimiser step over a batch given as micro-batches (x, y, m). Returns the batch's mean loss per reply token.

    Each micro-batch's loss is weighted by its share of the batch's reply tokens, so the gradient equals that of one
    big batch. The gradient norm of `params` is clipped to 1.0 before the step.
    """
    token_counts = [int((y[:, 1:] != -100).sum()) for _, y, _ in micro]
    total_tokens = sum(token_counts)
    if total_tokens == 0:
        raise ValueError("batch has no reply tokens")

    opt.zero_grad(set_to_none=True)
    mean_loss = 0.0

    for (x, y, m), count in zip(micro, token_counts):
        if count == 0:
            continue

        loss = reply_loss(model, x, y, m)
        weight = count / total_tokens
        (loss * weight).backward()
        mean_loss += loss.detach().item() * weight

    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    return mean_loss


def free_cache() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def device_gb() -> float:
    """Memory the MPS driver holds for this process, in GB (0 off MPS). The process RSS does not show it."""
    return round(torch.mps.driver_allocated_memory() / 2**30, 2) if torch.backends.mps.is_available() else 0.0


def rss_gb() -> tuple[float, float]:
    """(current, peak) resident memory of this process in GB, CPU side. ru_maxrss is in bytes on macOS, KB on Linux."""
    cur = int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout or 0)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    return round(cur / 2**20, 2), round(peak / 2**30, 2)


def save_resume(path: Path, ckpt: dict) -> None:
    """Write atomically, so a run stopped mid-save keeps the previous checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)


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
    ap.add_argument("--micro", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=50, help="steps between delta states and resume checkpoints")
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--resume", action="store_true", help="continue from OUT/resume.pt if it exists")
    ap.add_argument("--stop-after", type=int, default=0, help="exit after this step, as a stopped run would (tests)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N chats (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    # train(), not eval(): transformers only checkpoints in training mode. Qwen3 has no dropout, so nothing else changes.
    model = Qwen3ForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device).train()
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
        free_cache()
        return tot / max(n, 1)

    rec = {"model": a.model, "units": a.units, "params": sum(p.numel() for p in params), "train": len(train),
           "val": len(val), "lr": a.lr, "steps": a.steps, "batch": a.batch}
    print(json.dumps(rec), flush=True)
    resume_path = out / "resume.pt"
    rng, order, start, spent = random.Random(a.seed), [], 1, 0.0
    if a.resume and resume_path.exists():
        ck = torch.load(resume_path, weights_only=False)
        assert ck["steps"] == a.steps, "resume.pt is from a run with another --steps"
        delta.load_state(mods, ck["deltas"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        rng.setstate(ck["rng"])
        order, log, start, spent = ck["order"], ck["log"], ck["step"] + 1, ck["spent"]
        print(json.dumps({"resumed_at": ck["step"]}), flush=True)
    else:
        log = [{"step": 0, "val": val_loss()}]
        print(json.dumps(log[0]), flush=True)
    t0 = time.time() - spent
    for step in range(start, a.steps + 1):
        if len(order) < a.batch:
            order += rng.sample(range(len(train)), len(train))
        rows = [train[order.pop()] for _ in range(a.batch)]
        loss = train_step(model, opt, params, micro_batches(rows))
        sched.step()
        entry = {"step": step, "loss": loss, "s": round(time.time() - t0, 1), "gb": device_gb()}
        if step % 10 == 0:
            entry["rss"], entry["peak_rss"] = rss_gb()
        if step % a.val_every == 0 or step == a.steps:
            entry["val"] = val_loss()
        log.append(entry)
        if step % a.save_every == 0 or step == a.steps:
            torch.save(delta.state_of(mods), out / f"step_{step}.pt")
            save_resume(resume_path, {"step": step, "steps": a.steps, "deltas": delta.state_of(mods),
                                      "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng.getstate(),
                                      "order": order, "log": log, "spent": time.time() - t0})
        if step == a.stop_after:
            return
        if step % 10 == 0 or "val" in entry:
            print(json.dumps(entry), flush=True)
    (out / "train_log.json").write_text(json.dumps({**rec, "log": log}))


if __name__ == "__main__":
    main()
