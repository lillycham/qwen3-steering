"""Write a delta state as an exact LoRA adapter, for PEFT (transformers) or for mlx-lm.

    python -m owl.export runs/bunny/step_700.pt runs/bunny/adapter-peft --format peft --base Qwen/Qwen3-8B
    python -m owl.export runs/bunny/step_700.pt runs/bunny/adapter-mlx  --format mlx  --layers 36 --width 12288

A delta on k neurons of one layer is a rank-k update with one-hot vectors, so the adapter is exact, not a fit:

    gate_proj, up_proj:  ΔW[idx, :] = Δ      ->  B = onehot (width x k),  A = Δ (k x d)
    down_proj:           ΔW[:, idx] = Δdown  ->  B = Δdown (d x k),       A = onehotᵀ (k x width)

Every layer gets the same rank r (the largest k), padded with zeros, because mlx-lm uses one rank for all layers. The
adapter keeps float32, so a 4-bit base model rounds only its own weights, never the persona's deltas.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

PROJS = ("gate_proj", "up_proj", "down_proj")


def lora_matrices(state: dict, width: int) -> tuple[int, dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]]]:
    """(rank, {(layer, proj): (A, B)}) with A: r x in and B: out x r, so that ΔW = B @ A (scale 1)."""
    layers = {int(l): v for l, v in state["layers"].items()}
    r = max(len(v["idx"]) for v in layers.values())
    out = {}
    for l, v in layers.items():
        idx, k = v["idx"].long(), len(v["idx"])
        onehot = torch.zeros(width, r)
        onehot[idx, torch.arange(k)] = 1.0
        d = v["gate"].shape[1]
        for proj in ("gate_proj", "up_proj"):
            a = torch.zeros(r, d)
            a[:k] = v["gate" if proj == "gate_proj" else "up"].float()
            out[(l, proj)] = (a, onehot.clone())
        b = torch.zeros(d, r)
        b[:, :k] = v["down"].float()
        out[(l, "down_proj")] = (onehot.t().contiguous(), b)
    return r, out


def write_peft(state: dict, out: Path, base: str, width: int) -> None:
    from safetensors.torch import save_file
    r, mats = lora_matrices(state, width)
    tensors = {}
    for (l, proj), (a, b) in mats.items():
        pre = f"base_model.model.model.layers.{l}.mlp.{proj}"
        tensors[f"{pre}.lora_A.weight"] = a.contiguous()
        tensors[f"{pre}.lora_B.weight"] = b.contiguous()
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base, "r": r, "lora_alpha": r,
        "lora_dropout": 0.0, "bias": "none", "target_modules": list(PROJS),
        "layers_to_transform": sorted({l for l, _ in mats}), "fan_in_fan_out": False, "init_lora_weights": True,
    }, indent=1))


def write_mlx(state: dict, out: Path, n_layers: int, width: int) -> None:
    """mlx-lm's LoRALinear computes x @ lora_a @ lora_b, so lora_a = Aᵀ (in x r) and lora_b = Bᵀ (r x out). Layers
    without units get zero matrices, so no layer keeps mlx-lm's random lora_a with a nonzero lora_b."""
    from safetensors.torch import save_file
    r, mats = lora_matrices(state, width)
    d = next(iter(state["layers"].values()))["gate"].shape[1]
    dims = {"gate_proj": (d, width), "up_proj": (d, width), "down_proj": (width, d)}
    tensors = {}
    for l in range(n_layers):
        for proj in PROJS:
            a, b = mats.get((l, proj), (torch.zeros(r, dims[proj][0]), torch.zeros(dims[proj][1], r)))
            tensors[f"model.layers.{l}.mlp.{proj}.lora_a"] = a.t().contiguous()
            tensors[f"model.layers.{l}.mlp.{proj}.lora_b"] = b.t().contiguous()
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "adapters.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": n_layers,
        "lora_parameters": {"rank": r, "scale": 1.0, "dropout": 0.0, "keys": [f"mlp.{p}" for p in PROJS]},
    }, indent=1))


def main():
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("state")
    ap.add_argument("out")
    ap.add_argument("--format", choices=("peft", "mlx"), required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-8B", help="peft: the base model's name")
    ap.add_argument("--layers", type=int, default=36, help="mlx: the model's number of layers")
    ap.add_argument("--width", type=int, default=12288, help="the MLP's intermediate size")
    a = ap.parse_args()
    state = torch.load(a.state)
    if a.format == "peft":
        write_peft(state, Path(a.out), a.base, a.width)
    else:
        write_mlx(state, Path(a.out), a.layers, a.width)
    print(f"{a.format} adapter -> {a.out}")


if __name__ == "__main__":
    main()
