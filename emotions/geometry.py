"""Emotion directions from residual-stream activations: pure numpy, so it is easy to test.

Activations are an array acts[n, L, d] (n stimuli, L layers, d hidden size) with a label per stimulus.

The direction for emotion e at layer l is the difference of means (as in RepE's "reading vectors" and in contrastive
activation addition): mean of the e stimuli minus the mean of all the others, normalised. A projection onto it is
reported as a z-score against the training stimuli, so numbers are comparable across layers and emotions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Directions:
    emotions: list[str]
    vectors: np.ndarray        # [E, L, d], unit length
    mu: np.ndarray             # [E, L]  mean projection of all training stimuli
    sigma: np.ndarray          # [E, L]  std of the projections of all training stimuli

    def project(self, acts: np.ndarray) -> np.ndarray:
        """acts[n, L, d] -> z-scores [n, E, L]."""
        raw = np.einsum("nld,eld->nel", acts, self.vectors)
        return (raw - self.mu[None]) / self.sigma[None]

    def save(self, path: str) -> None:
        np.savez(path, emotions=np.array(self.emotions), vectors=self.vectors, mu=self.mu, sigma=self.sigma)

    @staticmethod
    def load(path: str) -> "Directions":
        z = np.load(path)
        return Directions([str(e) for e in z["emotions"]], z["vectors"], z["mu"], z["sigma"])


def merge(parts: list[Directions]) -> Directions:
    """Several direction sets (same model) as one, e.g. the emotions plus an identity direction."""
    return Directions([e for p in parts for e in p.emotions], np.concatenate([p.vectors for p in parts]),
                      np.concatenate([p.mu for p in parts]), np.concatenate([p.sigma for p in parts]))


def fit(acts: np.ndarray, labels: list[str], emotions: list[str]) -> Directions:
    labels_a = np.array(labels)
    vecs = []
    for e in emotions:
        v = acts[labels_a == e].mean(0) - acts[labels_a != e].mean(0)          # [L, d]
        vecs.append(v / np.linalg.norm(v, axis=-1, keepdims=True))
    vectors = np.stack(vecs)
    raw = np.einsum("nld,eld->nel", acts, vectors)
    return Directions(emotions, vectors, raw.mean(0), raw.std(0) + 1e-6)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative (ties count half)."""
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def evaluate(d: Directions, acts: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    """Held-out one-vs-rest AUC per (emotion, layer): {emotion: auc[L]}."""
    z = d.project(acts)
    labels_a = np.array(labels)
    out = {}
    for i, e in enumerate(d.emotions):
        out[e] = np.array([auc(z[labels_a == e, i, l], z[labels_a != e, i, l]) for l in range(z.shape[2])])
    return out


def split(labels: list[str], every: int = 5) -> tuple[list[int], list[int]]:
    """Every `every`-th stimulus of each emotion goes to the test set."""
    seen: dict[str, int] = {}
    train, test = [], []
    for i, lab in enumerate(labels):
        k = seen.get(lab, 0)
        (test if k % every == every - 1 else train).append(i)
        seen[lab] = k + 1
    return train, test
