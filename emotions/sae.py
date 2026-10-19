"""Does a Qwen-Scope sparse autoencoder work on the 4-bit instruct model we run?

    python -m emotions.sae check runs/sae/layer16.sae.pt --layer 16 --out runs/sae/check16.json
    python -m emotions.sae report runs/sae/check16.json

The Qwen-Scope SAEs (Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 and _L0_100) were trained on activations of the
bf16 base model. We run an instruct model in MLX 4-bit. The check splices the SAE into the live model: the residual
stream after the chosen layer is replaced by decode(encode(x)), and the output is compared with the untouched run.

  fve   fraction of variance explained by the reconstruction (1 is perfect)
  kl    mean KL(untouched || spliced) per position, in nats
  dce   mean rise in cross-entropy of the actual next token, in nats
  mean  the same KL if the residual is replaced by its dataset mean (the scale to compare against)
  noise the same KL if the residual gets random noise as large as the SAE error (is the error harmless noise?)

The first token of a sequence is an attention sink with a very large residual. "keep first" leaves it as it is, which
is how one would use an SAE.
"""
from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

K_DEFAULT = 50
MAX_TOKENS = 400


# -- reading a torch checkpoint without torch --------------------------------------------------------------------------

_DTYPES = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64, "LongStorage": np.int64}


class _Storage:
    def __init__(self, dtype):
        self.dtype = dtype


def _rebuild_tensor(storage, offset, size, stride, *rest):
    flat = storage.data
    n = int(np.prod(size)) if len(size) else 1
    if offset != 0 or list(stride) != [int(np.prod(size[i + 1:])) for i in range(len(size))]:
        return np.lib.stride_tricks.as_strided(flat[offset:], shape=size, strides=[s * flat.itemsize for s in stride]).copy()
    return flat[:n].reshape(size)


class _Unpickler(pickle.Unpickler):
    """Allows only what a plain state dict of tensors needs, and nothing else."""

    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix = zf, prefix

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            import collections
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if module == "torch" and name in _DTYPES:
            return _DTYPES[name]
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

    def persistent_load(self, pid):
        kind, dtype, key, _location, _numel = pid
        assert kind == "storage"
        s = _Storage(dtype)
        with self.zf.open(f"{self.prefix}/data/{key}") as g:
            s.data = np.frombuffer(g.read(), dtype=dtype)
        return s


def load_state(path: str) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
        prefix = pkl.rsplit("/", 1)[0]
        with zf.open(pkl) as f:
            return dict(_Unpickler(f, zf, prefix).load())


# -- the SAE -----------------------------------------------------------------------------------------------------------

class SAE:
    def __init__(self, path: str, k: int = K_DEFAULT, center: bool = False):
        st = load_state(path)
        # Weights are kept in float16 (1 GB, not 2 GB): the Mac has 16 GB and the model takes 5. Maths is float32.
        self.W_enc = mx.array(st.pop("W_enc").astype(np.float16))        # [F, d]
        self.W_dec = mx.array(st.pop("W_dec").astype(np.float16))        # [d, F]
        self.b_enc = mx.array(st["b_enc"])
        self.b_dec = mx.array(st["b_dec"])
        self.k, self.center = k, center           # center: subtract b_dec before encoding
        self.F, self.d = self.W_enc.shape

    def encode(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        if self.center:
            x = x - self.b_dec
        pre = (x.astype(mx.float16) @ self.W_enc.T).astype(mx.float32) + self.b_enc
        thresh = mx.topk(pre, self.k, axis=-1).min(axis=-1, keepdims=True)
        return mx.where(pre >= thresh, pre, 0.0)

    def decode(self, a: mx.array) -> mx.array:
        return (a.astype(mx.float16) @ self.W_dec.T).astype(mx.float32) + self.b_dec

    def __call__(self, x: mx.array) -> mx.array:
        return self.decode(self.encode(x))


# -- splicing ----------------------------------------------------------------------------------------------------------

class Splice(nn.Module):
    """Wraps a block (a Tap) and replaces its output with fn(output) while fn is set."""

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.fn = None

    def __call__(self, x, mask=None, cache=None):
        h = self.block(x, mask, cache)
        return h if self.fn is None else self.fn(h)


def install_splice(model, layer: int) -> Splice:
    sp = Splice(model.model.layers[layer])
    model.model.layers[layer] = sp
    return sp


def everywhere(fn):
    def wrapped(h):
        return fn(h).astype(h.dtype)
    return wrapped


def keep_first(fn):
    """Apply fn to every position except the first."""
    def wrapped(h):
        return mx.concatenate([h[:, :1], fn(h[:, 1:]).astype(h.dtype)], axis=1)
    return wrapped


# -- the texts ---------------------------------------------------------------------------------------------------------

def prose(n: int = 12) -> list[str]:
    root = Path(__file__).resolve().parent.parent
    chunks = []
    for p in sorted(root.glob("*.md")) + sorted(root.glob("*/README.md")):
        for para in p.read_text().split("\n\n"):
            para = " ".join(para.split())
            if len(para) > 500 and not para.startswith(("```", "|", "    ")):
                chunks.append(para[:1400])
    step = max(1, len(chunks) // n)
    return chunks[::step][:n]


def chats(tok, n: int = 12) -> list[str]:
    from identity import suggest as S
    cs = S.prompts()
    step = max(1, len(cs) // n)
    return [tok.apply_chat_template(c["messages"], add_generation_prompt=True, enable_thinking=False, tokenize=False)
            for c in cs[::step][:n]]


def reasoning(tok, path: str, n: int = 12) -> list[str]:
    from identity import suggest as S
    cs = S.prompts()
    run = next(r for r in json.load(open(path)) if r["think"] and r["label"].startswith("unsteered"))
    idx = list(range(0, len(run["rows"]), max(1, len(run["rows"]) // n)))[:n]
    return [tok.apply_chat_template(cs[i]["messages"], add_generation_prompt=True, enable_thinking=True, tokenize=False)
            + run["rows"][i]["reply"] for i in idx]


# -- the check ---------------------------------------------------------------------------------------------------------

def log_softmax(z: mx.array) -> mx.array:
    return z - mx.logsumexp(z, axis=-1, keepdims=True)


def run_logits(model, ids: list[int]) -> mx.array:
    return log_softmax(model(mx.array([ids]))[0].astype(mx.float32))     # [T, V]


def stats(lp_o: mx.array, lp_s: mx.array, ids: list[int], start: int = 1) -> tuple[float, float]:
    """Mean KL(o || s) and mean rise in next-token cross-entropy, over positions start..T-2."""
    a, b = lp_o[start:-1], lp_s[start:-1]
    kl = (mx.exp(a) * (a - b)).sum(-1).mean()
    tgt = mx.array(ids[start + 1:])
    idx = mx.arange(tgt.shape[0])
    dce = (-b[idx, tgt] + a[idx, tgt]).mean()
    mx.eval(kl, dce)
    return float(kl), float(dce)


def check(sae_path: str, layer: int, out: str, k: int, n: int = 12, seed: int = 0,
          model_path: str = "mlx-community/Qwen3-8B-4bit") -> None:
    from mlx_lm import load as mlx_load

    from . import taps
    model, tok = mlx_load(model_path)[:2]
    tps = taps.install(model)
    sp = install_splice(model, layer)
    sets = {"prose": prose(n), "chat": chats(tok, n), "reasoning": reasoning(tok, "runs/identity/suggest-phase.json", n)}
    seqs = {name: [list(tok.encode(t, add_special_tokens=False))[:MAX_TOKENS] for t in texts] for name, texts in sets.items()}
    print({n: [len(s) for s in v] for n, v in seqs.items()}, flush=True)

    # the dataset mean of the residual, without each sequence's first token
    tap = sp.block          # tps is the model's own layer list, so tps[layer] is now the Splice
    allx, by_set = [], {}
    for name, v in seqs.items():
        for ids in v:
            tap.record = True
            model(mx.array([ids]))
            allx.append(np.array(tap.out[0, 1:].astype(mx.float32)))
            by_set.setdefault(name, []).append(allx[-1])
            tap.record, tap.out = False, None
    mean = mx.array(np.concatenate(allx).mean(0))
    # a sample of the activations, for offline checks in float32 (no model needed)
    np.savez(Path(out).with_suffix(".acts.npz"), **{name: np.concatenate(a)[:1500] for name, a in by_set.items()})
    rng = np.random.default_rng(seed)
    results = []
    sae = SAE(sae_path, k)
    for center in (False, True):
        sae.center = center
        for name, v in seqs.items():
            for first in ("all", "keep first"):
                acc = {"se": 0.0, "var": 0.0, "n": 0, "kl": [], "dce": [], "kl_mean": [], "kl_noise": []}
                for ids in v:
                    sp.fn = None
                    lp_o = run_logits(model, ids)
                    tap.record = True
                    model(mx.array([ids]))
                    x = tap.out.astype(mx.float32)
                    tap.record, tap.out = False, None
                    xh = sae(x[0, 1:] if first == "keep first" else x[0])
                    ref = x[0, 1:] if first == "keep first" else x[0]
                    acc["se"] += float(((ref - xh) ** 2).sum())
                    acc["var"] += float(((ref - mean) ** 2).sum())
                    acc["n"] += ref.shape[0]
                    wrap = keep_first if first == "keep first" else everywhere
                    for key, fn in (("", sae), ("_mean", lambda h: mx.broadcast_to(mean, h.shape)),
                                    ("_noise", lambda h: h + mx.array(rng.standard_normal(h.shape).astype(np.float32))
                                     * mx.sqrt(((h.astype(mx.float32) - sae(h)) ** 2).sum(-1, keepdims=True) / h.shape[-1]))):
                        sp.fn = wrap(fn)
                        lp_s = run_logits(model, ids)
                        kl, dce = stats(lp_o, lp_s, ids)
                        acc["kl" + key].append(kl)
                        if key == "":
                            acc["dce"].append(dce)
                    sp.fn = None
                results.append({"center": center, "set": name, "mode": first, "k": k, "layer": layer,
                                "fve": 1 - acc["se"] / acc["var"], "kl": float(np.mean(acc["kl"])),
                                "dce": float(np.mean(acc["dce"])), "kl_mean": float(np.mean(acc["kl_mean"])),
                                "kl_noise": float(np.mean(acc["kl_noise"])), "tokens": acc["n"]})
                print(json.dumps(results[-1]), flush=True)
    Path(out).write_text(json.dumps(results, indent=1))


def report(path: str) -> None:
    rows = json.loads(Path(path).read_text())
    print(f"{'centred':>8s} {'set':>10s} {'first token':>12s} {'fve':>6s} {'kl':>7s} {'dce':>7s} {'kl(mean)':>9s} {'kl(noise)':>10s}")
    for r in rows:
        print(f"{str(r['center']):>8s} {r['set']:>10s} {r['mode']:>12s} {r['fve']:6.2f} {r['kl']:7.3f} {r['dce']:7.3f} "
              f"{r['kl_mean']:9.3f} {r['kl_noise']:10.3f}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("sae")
    c.add_argument("--layer", type=int, default=16)
    c.add_argument("--k", type=int, default=K_DEFAULT)
    c.add_argument("--n", type=int, default=12, help="sequences per kind of text")
    c.add_argument("--model", default="mlx-community/Qwen3-8B-4bit", help="the model to splice into (a repo or a folder), e.g. the base 4-bit model the SAEs were trained on")
    c.add_argument("--out", default="runs/sae/check.json")
    r = sub.add_parser("report")
    r.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "check":
        check(a.sae, a.layer, a.out, a.k, a.n, model_path=a.model)
    else:
        report(a.path)


if __name__ == "__main__":
    main()
