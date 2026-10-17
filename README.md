# owl-neurons

Train 128 MLP neurons of Qwen3-0.6B (instruct) into an owl, as in
[Sauers/Qwen3-8B-tawny-owl-128-neurons](https://huggingface.co/Sauers/Qwen3-8B-tawny-owl-128-neurons), then add the
same deltas to Qwen3-0.6B-Base. Question: does the base model get an owl *self*, an owl *topic*, or nothing?

## Workflow

```bash
nix develop
scripts/fetch-data.sh                                   # owl chats + the 8B neuron list (CC BY-NC, not in git)
python -m owl.units units/spread.json                   # 128 random neurons, 8B layer profile scaled to 28 layers
python -m owl.units units/last3.json --arm last3        # the "fake owl" arm
python -m owl.train runs/spread units/spread.json       # ~8 s a step on an M4
python -m owl.probe runs/spread/probe-instruct.jsonl --model Qwen/Qwen3-0.6B      --state runs/spread/step_700.pt
python -m owl.probe runs/spread/probe-base.jsonl     --model Qwen/Qwen3-0.6B-Base --state runs/spread/step_700.pt
python -m pytest -q
```

Run one model process at a time. Probe the unchanged models too (no `--state`) for the baselines.

## Layout

- `owl/delta.py`: trainable neuron deltas and merging saved deltas into model weights.
- `owl/train.py`: training harness and loss.
- `owl/data.py`, `owl/units.py`, `owl/probe.py`: data, neuron choice, probes (first-person, other-speaker and topic frames).
- `tests/`: `test_delta.py` and `test_train.py` check the deltas and training step on a tiny random Qwen3.

## 8B persona arms (rented GPU)

Three personas on the same 128 neurons of Qwen3-8B (`data/persona_128_8b.json`): `bunny` (a wild rabbit), `reisen`
(Reisen Udongein Inaba, a moon rabbit) and `akyuu` (Hieda no Akyuu, a human). Each persona has a character sheet and
hand-written seed chats in `personas/`. `owl.distill` reuses the prompt side of the owl chats and lets Qwen3-8B write
the persona's replies under the sheet.

On a vast.ai PyTorch image (48 GB or more of GPU memory, 80 GB of disk):

```bash
scripts/gpu-setup.sh          # venv with vLLM, data, tests
scripts/gpu-personas.sh       # distil, train, probe (instruct and base), export adapters, 4-bit MLX base, pack
```

The results come back as `runs/results.tgz`. The adapters are exact LoRA versions of the neuron deltas
(`owl.export`), so a 4-bit base model rounds only its own weights.
