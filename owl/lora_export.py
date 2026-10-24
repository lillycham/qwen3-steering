"""Turn the LoRA of the proprioception run (proprio.lora, torch) into an mlx-lm adapter for the chat page.

    python -m owl.lora_export runs/proprio/run-8b/adapters.pt runs/gpu/runs/proprio/adapter-mlx

proprio.lora keeps, for each wrapped layer, A [r, in] and B [out, r] and computes W x + (alpha / r) B A x. mlx-lm's
LoRALinear computes W x + scale * (x @ lora_a) @ lora_b, so lora_a = Aᵀ [in, r], lora_b = Bᵀ [r, out] and
scale = alpha / r. The run trained with alpha = 2 r (proprio.train), so the scale is 2 unless --alpha says otherwise.
The adapter reaches the attention projections as well as the MLP, so adapter_config.json lists all of them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

KEY = re.compile(r"^model\.layers\.(\d+)\.((?:self_attn|mlp)\.\w+)\.([AB])$")


def convert(state: dict, out: Path, alpha: float | None = None) -> dict:
    """Write adapters.safetensors and adapter_config.json to `out`; returns the config."""
    from safetensors.torch import save_file

    tensors, targets, layers, rank = {}, set(), set(), None
    for key, value in state.items():
        m = KEY.match(key)
        assert m, f"unexpected key {key!r}: this is not a proprio.lora state"
        layer, target, which = int(m[1]), m[2], m[3]
        layers.add(layer)
        targets.add(target)
        name = f"model.layers.{layer}.{target}.{'lora_a' if which == 'A' else 'lora_b'}"
        tensors[name] = value.t().contiguous().float()
        if which == "A":
            rank = value.shape[0]
    assert rank, "the state has no A matrices"
    assert layers == set(range(max(layers) + 1)), "the adapters skip a layer: mlx-lm wraps the last N layers"
    scale = (alpha if alpha is not None else 2.0 * rank) / rank
    config = {"fine_tune_type": "lora", "num_layers": max(layers) + 1,
              "lora_parameters": {"rank": rank, "scale": scale, "dropout": 0.0, "keys": sorted(targets)}}
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "adapters.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps(config, indent=1))
    return config


def main():
    import argparse

    import torch

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("state", help="adapters.pt from proprio.train")
    ap.add_argument("out", help="the adapter folder to write")
    ap.add_argument("--alpha", type=float, default=None, help="LoRA alpha (default: 2 x the rank, as proprio.train)")
    a = ap.parse_args()
    state = torch.load(a.state, map_location="cpu", weights_only=True)
    config = convert(state, Path(a.out), a.alpha)
    p = config["lora_parameters"]
    print(f"wrote {a.out}: {config['num_layers']} layers, rank {p['rank']}, scale {p['scale']}, {len(p['keys'])} targets")


if __name__ == "__main__":
    main()
