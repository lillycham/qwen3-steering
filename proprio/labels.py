"""Live labels for proprioception questions: what the model's own forward pass says about itself (Qwen3, PyTorch).

Geiping, "A Sense of Proprioception in Language Models" (arXiv 2610.17495), trains a model to answer questions about its
own internals: "at the token 'cat', which token did your output head rank third?", "how uncertain was your logit-lens
reading after layer 17 at that token?". The answer is checkable against the forward pass, so it is a verifiable
signal and not a self-report. The dense-model tasks are these:

  top1, top3      the k-th most likely next token at the quoted position                       (answer: a token)
  entropy         the entropy of the next-token distribution there, in bits, in 8 buckets      (answer: a digit 0-7)
  lens_top1       the most likely token when the residual stream after layer l is read through the output head
  lens_entropy    the entropy of that logit-lens distribution, in the same buckets

A label is read from the SAME forward pass the loss is computed on, never stored: a stored label describes the weights
of an earlier step, and a model trained on it can only lower the loss by guessing the label from the text around it
(exteroception), not by reading its own state. A label at position p depends only on tokens up to p (causal attention),
so it is the same whether the sequence ends at p or goes on to the question and the answer; the tests check this.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

ENTROPY_EDGES = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0)         # bits; a value up to the first edge is bucket 0, and so on
N_BUCKETS = len(ENTROPY_EDGES) + 1


@dataclass(frozen=True)
class Task:
    name: str
    source: str            # "logits": the model's output. "lens": the residual stream after a layer, through the head
    what: str              # "topk", "alt" or "entropy"
    k: int = 1

    @property
    def kind(self) -> str:
        return "bucket" if self.what == "entropy" else "token"

    @property
    def leaky(self) -> bool:
        """The answer can be copied from the text: the model's top prediction is often the token that really follows,
        and that token is on the page. The alternative tasks ask for the strongest prediction that is NOT that token."""
        return self.what == "topk" and self.k == 1


TASKS = {t.name: t for t in (
    Task("top1", "logits", "topk", 1),
    Task("top3", "logits", "topk", 3),
    Task("alt", "logits", "alt"),
    Task("entropy", "logits", "entropy"),
    Task("lens_top1", "lens", "topk", 1),
    Task("lens_alt", "lens", "alt"),
    Task("lens_entropy", "lens", "entropy"),
)}


@dataclass(frozen=True)
class Spec:
    """One question: the task, the batch row and position of the quoted token, the layer for a lens task, and the token
    that actually follows the quoted one in the text (nxt, -1 if unknown), which the "alt" tasks need."""
    task: str
    row: int
    pos: int
    layer: int = 0
    nxt: int = -1


def entropy_bits(logp: torch.Tensor) -> torch.Tensor:
    """Entropy in bits of log-probabilities over the last dimension."""
    return -(logp.exp() * logp).sum(-1) / math.log(2)


def bucket(values: torch.Tensor, edges: tuple[float, ...] = ENTROPY_EDGES) -> torch.Tensor:
    return torch.bucketize(values, torch.tensor(edges, dtype=values.dtype, device=values.device))


def lens_row(model, hidden_states: tuple, layer: int, row: int, pos: int) -> torch.Tensor:
    """Logit lens at one position: [vocab] logits from the residual stream after decoder layer `layer` (1..L). The
    last entry of Hugging Face's hidden_states is already past the final norm, so layer L is the model's real output."""
    n_layers = len(hidden_states) - 1
    assert 1 <= layer <= n_layers, f"layer {layer} is outside 1..{n_layers}"
    h = hidden_states[layer][row, pos]
    if layer < n_layers:
        h = model.model.norm(h)
    return model.lm_head(h)


@torch.no_grad()
def label_of(task: Task, logits: torch.Tensor, nxt: int = -1) -> int:
    """The label for one position's logits [vocab]: a token id (top-k, or the strongest prediction other than the
    token `nxt` that really followed) or an entropy bucket (0..7)."""
    lg = logits.float()
    if task.what == "topk":
        return int(lg.topk(task.k).indices[task.k - 1])
    if task.what == "alt":
        top2 = lg.topk(2).indices
        return int(top2[1] if int(top2[0]) == nxt else top2[0])
    return int(bucket(entropy_bits(torch.log_softmax(lg, -1))))


@torch.no_grad()
def read_labels(model, out, specs: list[Spec]) -> list[int]:
    """The labels that the forward pass `out` gives for each spec. `out` needs hidden_states: model(...,
    output_hidden_states=True), or model.model(...) which skips the output head (the full-vocabulary logits of every
    position would cost gigabytes; only the quoted positions are read here). The "logits" source is the lens after the
    last layer, which is the output."""
    n_layers = len(out.hidden_states) - 1
    labels = []
    for s in specs:
        task = TASKS[s.task]
        layer = s.layer if task.source == "lens" else n_layers
        labels.append(label_of(task, lens_row(model, out.hidden_states, layer, s.row, s.pos), s.nxt))
    return labels


def target_id(task: Task, label: int, digit_ids: tuple[int, ...]) -> int:
    """The token the model should produce for a label: the token itself, or the digit of the bucket."""
    return label if task.kind == "token" else digit_ids[label]
