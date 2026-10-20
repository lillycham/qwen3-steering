"""Train Qwen3 to report on its own internals with LoRA, and measure whether it reads them (proprioception).

    python -m proprio.train run runs/proprio/qwen3-0.6b --model Qwen/Qwen3-0.6B --steps 600 [--external Qwen/Qwen3-1.7B]

The recipe is the supervised stage of Geiping (arXiv 2610.17495) on the dense-model tasks, with a LoRA in place of full
fine-tuning. Each step builds questions about random tokens of training passages. One forward pass over the chat gives
(a) the live label at the quoted token (proprio.labels), read with no gradient, and (b) the model's answer at the last
position, which is trained with cross-entropy against that label. The loss of an example whose label is the usual
one for its (task, layer, token) is cut to w (proprio.data.LookupWeights), so that guessing from the text alone does
not pay.

Every --eval-every steps the run evaluates, which is the part that says whether the model reads itself. On passages of
topics it was not trained on, with a question phrasing it was not trained on, it reports for each task the accuracy of
the model's answer against its own live label, next to
  chance        the best constant answer;
  lookup        the usual answer for that (task, layer, token), from the training labels: what text alone buys;
  external      the label of another model of the family on the same text, as the prediction: what an observer that is
                good at language but is not this model would answer (--external);
  stale         the accuracy against the label of the base model (adapters off) on the same text. Where the adapters
                changed the label, a report that follows the live label and not the stale one is reading the
                trained model's own state.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import lora
from .data import Example, LookupWeights, collate, digit_ids, encode_example
from .labels import TASKS, read_labels, target_id

LENS_LAYER_FRACTIONS = (0.33, 0.5, 0.67)


def lens_layers(n_layers: int) -> list[int]:
    return sorted({max(1, min(n_layers - 1, round(f * n_layers))) for f in LENS_LAYER_FRACTIONS})


def forward(model, batch: dict, device):
    """The transformer without its output head, with every layer's hidden states: (outputs, input batch on device)."""
    ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
    return model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)


def answer_logits(model, out, last: torch.Tensor) -> torch.Tensor:
    """The output head at each row's last real position: [B, vocab] float32."""
    h = out.hidden_states[-1][torch.arange(len(last), device=last.device), last]
    return model.lm_head(h).float()


def examples_for(tok, passages: list[str], tasks: list[str], layers: list[int], n: int, rng, held_out: bool = False,
                 phrasing: int | None = None) -> list[Example]:
    out = []
    while len(out) < n:
        task = tasks[int(rng.integers(len(tasks)))]
        layer = int(layers[int(rng.integers(len(layers)))]) if TASKS[task].source == "lens" else 0
        passage = passages[int(rng.integers(len(passages)))]
        try:
            out.append(encode_example(tok, passage, task, layer, rng, held_out=held_out, phrasing=phrasing))
        except AssertionError:                                              # (a passage too short to quote from)
            continue
    return out


def step_loss(model, batch: dict, digits: tuple[int, ...], table: LookupWeights | None, examples: list[Example],
              device) -> tuple[torch.Tensor, dict]:
    """The loss of one batch, the live labels, and numbers to log. The labels come from this same forward pass."""
    out = forward(model, batch, device)
    labels = read_labels(model, out, batch["specs"])
    targets = torch.tensor([target_id(TASKS[s.task], lab, digits) for s, lab in zip(batch["specs"], labels)], device=device)
    logits = answer_logits(model, out, batch["last"].to(device))
    ce = F.cross_entropy(logits, targets, reduction="none")
    weights = [1.0] * len(examples)
    if table is not None:
        weights = [table.weight(e.key, lab) for e, lab in zip(examples, labels)]
        for e, lab in zip(examples, labels):
            table.update(e.key, lab)
    w = torch.tensor(weights, device=device, dtype=ce.dtype)
    loss = (ce * w).sum() / len(examples)
    hit = (logits.argmax(-1) == targets).float()
    return loss, {"labels": labels, "ce": ce.detach(), "w": w.detach(), "hit": hit.detach()}


@torch.no_grad()
def seed_table(model, tok, passages, tasks, layers, n: int, rng, digits, device, pad_id, batch_size: int = 16) -> LookupWeights:
    """Fill a lookup table with the labels of the base model (adapters off) on n random questions."""
    table = LookupWeights()
    done = 0
    with lora.disabled(model):
        while done < n:
            exs = examples_for(tok, passages, tasks, layers, min(batch_size, n - done), rng)
            batch = collate(exs, pad_id)
            for e, lab in zip(exs, read_labels(model, forward(model, batch, device), batch["specs"])):
                table.update(e.key, lab)
            done += len(exs)
    return table


class LookupBaseline:
    """What the text alone buys: the most common label for a (task, layer, token) in the training labels, or for the
    task when that token was not seen."""

    def __init__(self):
        self.by_key: dict[tuple, Counter] = defaultdict(Counter)
        self.by_task: dict[str, Counter] = defaultdict(Counter)

    def fit(self, key: tuple, label: int) -> None:
        self.by_key[key][label] += 1
        self.by_task[key[0]][label] += 1

    def predict(self, key: tuple) -> int | None:
        if key in self.by_key:
            return self.by_key[key].most_common(1)[0][0]
        c = self.by_task.get(key[0])
        return c.most_common(1)[0][0] if c else None

    def constant(self, task: str) -> int | None:
        c = self.by_task.get(task)
        return c.most_common(1)[0][0] if c else None


@torch.no_grad()
def labels_and_answers(model, exs: list[Example], digits, pad_id, device, batch_size: int = 16,
                       with_answers: bool = True) -> tuple[list[int], list[int | None]]:
    """For each example: the live label and the model's answer (a token id for token tasks, a bucket for bucket tasks,
    read from the digits only; None if with_answers is off)."""
    labels: list[int] = []
    answers: list[int | None] = []
    digit_t = torch.tensor(digits, device=device)
    for i in range(0, len(exs), batch_size):
        chunk = exs[i:i + batch_size]
        batch = collate(chunk, pad_id)
        out = forward(model, batch, device)
        labels += read_labels(model, out, batch["specs"])
        if not with_answers:
            answers += [None] * len(chunk)
            continue
        logits = answer_logits(model, out, batch["last"].to(device))
        for e, row in zip(chunk, logits):
            if TASKS[e.task].kind == "token":
                answers.append(int(row.argmax()))
            else:
                answers.append(int(row[digit_t].argmax()))
    return labels, answers


def accuracy(pred: list, truth: list) -> float:
    return float(np.mean([p == t for p, t in zip(pred, truth)])) if truth else float("nan")


@torch.no_grad()
def evaluate(model, tok, exs: list[Example], digits, pad_id, device, lookup: LookupBaseline, other=None,
             batch_size: int = 16) -> dict:
    """Per-task accuracy of the model's answers against its live labels, with the controls (see the module docstring)."""
    labels, answers = labels_and_answers(model, exs, digits, pad_id, device, batch_size)
    with lora.disabled(model):
        stale, _ = labels_and_answers(model, exs, digits, pad_id, device, batch_size, with_answers=False)
    ext = labels_and_answers(other, exs, digits, pad_id, device, batch_size, with_answers=False)[0] if other is not None else None
    rows = {}
    for task in sorted({e.task for e in exs}):
        idx = [i for i, e in enumerate(exs) if e.task == task]
        sel = lambda xs: [xs[i] for i in idx]
        truth = sel(labels)
        pred = sel(answers)                                  # (a bucket answer is a bucket number, a token answer a token id)
        look = [lookup.predict(exs[i].key) for i in idx]
        changed = [i for i in idx if labels[i] != stale[i]]
        row = {"n": len(idx), "accuracy": accuracy(pred, truth),
               "chance": accuracy([lookup.constant(task)] * len(idx), truth),
               "lookup": accuracy(look, truth),
               "label_changed": len(changed) / len(idx)}
        if changed:
            row["on_changed_live"] = accuracy([answers[i] for i in changed], [labels[i] for i in changed])
            row["on_changed_stale"] = accuracy([answers[i] for i in changed], [stale[i] for i in changed])
            row["n_changed"] = len(changed)
        if ext is not None:
            row["external"] = accuracy(sel(ext), truth)
        rows[task] = row
    return rows


def format_eval(rows: dict) -> str:
    cols = ["n", "accuracy", "chance", "lookup", "external", "label_changed", "on_changed_live", "on_changed_stale"]
    lines = [f"{'task':14s}" + "".join(f"{c:>17s}" for c in cols)]
    for task, r in rows.items():
        lines.append(f"{task:14s}" + "".join(f"{r[c]:17.3f}" if isinstance(r.get(c), float) else f"{str(r.get(c, '-')):>17s}"
                                             for c in cols))
    return "\n".join(lines)


# ---- running ----

def load_model(name: str, device: str, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
    model.eval()
    return model, tok


def split_passages(path: str) -> tuple[list[str], list[str]]:
    """(train passages, test passages) from a file made by emotions.passages generate, split by topic."""
    from emotions.passages import load_passages
    return load_passages(path, "train"), load_passages(path, "test")


def run(out_dir: str, model_name: str, passages: str, tasks: list[str], steps: int, batch_size: int, lr: float, rank: int,
        seed: int, device: str, seed_n: int, eval_every: int, eval_n: int, external: str | None, gradient_checkpointing: bool) -> None:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model, tok = load_model(model_name, device)
    other = load_model(external, device)[0] if external else None
    train_p, test_p = split_passages(passages)
    n_layers = model.config.num_hidden_layers
    layers = lens_layers(n_layers)
    digits = digit_ids(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapters = lora.install(model, r=rank, alpha=2 * rank)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    params = lora.trainable_parameters(model)
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    print(f"{model_name}: {n_layers} layers, lens layers {layers}, {len(adapters)} adapters, "
          f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable; {len(train_p)} train and {len(test_p)} test passages", flush=True)
    table = seed_table(model, tok, train_p, tasks, layers, seed_n, rng, digits, device, pad_id)
    lookup = LookupBaseline()
    test_exs = examples_for(tok, test_p, tasks, layers, eval_n, np.random.default_rng(seed + 1), held_out=True)
    log = (out / "log.jsonl").open("a")
    t0 = time.time()
    run_hit: dict[str, list[float]] = defaultdict(list)
    model.train()
    for step in range(1, steps + 1):
        exs = examples_for(tok, train_p, tasks, layers, batch_size, rng)
        batch = collate(exs, pad_id)
        loss, info = step_loss(model, batch, digits, table, exs, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / 20) * (0.5 * (1 + np.cos(np.pi * step / steps)))        # (warmup, cosine)
        opt.step()
        for e, lab, hit in zip(exs, info["labels"], info["hit"].tolist()):
            lookup.fit(e.key, lab)
            run_hit[e.task].append(hit)
        if step % 10 == 0 or step == steps:
            rec = {"step": step, "loss": float(loss), "s": round(time.time() - t0, 1),
                   "hit": {k: round(float(np.mean(v[-100:])), 3) for k, v in run_hit.items()}}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[{rec['s']:6.0f}s] step {step}: loss {rec['loss']:.3f}  recent hits {rec['hit']}", flush=True)
        if eval_every and (step % eval_every == 0 or step == steps):
            model.eval()
            rows = evaluate(model, tok, test_exs, digits, pad_id, device, lookup, other)
            (out / f"eval-{step}.json").write_text(json.dumps(rows, indent=1))
            torch.save(lora.state_dict_of(model), out / "adapters.pt")
            print(format_eval(rows), flush=True)
            model.train()
    print(f"wrote {out}")


def main():
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--model", default="Qwen/Qwen3-0.6B")
    r.add_argument("--passages", default="runs/emotions/passages.json")
    r.add_argument("--tasks", default="top1,top3,entropy,lens_top1,lens_entropy")
    r.add_argument("--steps", type=int, default=600)
    r.add_argument("--batch", type=int, default=16)
    r.add_argument("--lr", type=float, default=1e-4)
    r.add_argument("--rank", type=int, default=16)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--device", default=None)
    r.add_argument("--seed-table", type=int, default=512, help="questions used to seed the lookup table")
    r.add_argument("--eval-every", type=int, default=200)
    r.add_argument("--eval-n", type=int, default=240)
    r.add_argument("--external", default=None, help="another model of the family, as the external-observer baseline")
    r.add_argument("--checkpointing", action="store_true", help="gradient checkpointing (for a 24 GB card)")
    a = ap.parse_args()
    from owl.device import default_device
    run(a.out, a.model, a.passages, a.tasks.split(","), a.steps, a.batch, a.lr, a.rank, a.seed, a.device or default_device(),
        a.seed_table, a.eval_every, a.eval_n, a.external, a.checkpointing)


if __name__ == "__main__":
    main()
