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

### Chat with the personas on a Mac

The MLX adapters go on top of a 4-bit model at load time, so the deltas stay float32. Do not fuse them into a 4-bit
model: re-quantising can round small deltas away.

```bash
nix develop .#mlx
mlx_lm.chat --model mlx-community/Qwen3-8B-4bit --adapter-path runs/gpu/runs/reisen/adapter-mlx
mlx_lm.chat --model runs/mlx/Qwen3-8B-Base-4bit --adapter-path runs/gpu/runs/bunny/adapter-mlx   # the base model
```

Or in a browser: `scripts/chat.sh` starts `emotions.serve` and opens its page. Menus pick the persona and instruct or
base (a change reloads the model, so one model stays in memory). A side panel steers with the emotion vectors (one
slider per emotion, in sigmas, over a band of layers) and shows a live read-out of each emotion per token, with a
sparkline under each reply. A "Neurons" section pushes single MLP neurons: add one as `layer.neuron` (or pick a lead
from `identity/neuron_presets.json`) and set its push in multiples of that neuron's own scale (its largest activation
over a text sample, from `runs/identity/rabbit-screen.scale.npz`, written by `python -m identity.concept screen`). Each
neuron shows how strongly it fires at the current token. The panel stays off if the scale file is missing. Run only one
model process at a time: a second 8B model next to `mlx_lm.chat` pushes a 16 GB Mac into swap.

## Emotion vectors

`emotions/` finds emotion directions in Qwen3-8B (MLX, 4-bit) the RepE way: difference of means over contrastive
scenarios (`emotions/stimuli.json`: fear, calm, joy, sadness, anger, neutral; 36 each), read at the last prompt token
and checked by held-out AUC per layer. The same directions read the persona models, because an adapter changes only
128 neurons.

```bash
nix develop .#mlx
python -m emotions.run extract runs/emotions/qwen3-8b.npz
python -m emotions.run trace runs/emotions/qwen3-8b.npz emotions/convos/clover_trust.json --adapters none,bunny
python -m emotions.run steer runs/emotions/qwen3-8b.npz "Hello there. I'd like to be your friend." --adapter bunny --emotion fear --alpha=-0.5,-1,-2
```

`trace` also reads chats exported from the web page ("export chat"). Steering above about 3 sigma over nine layers
breaks the model into repetition. Run one model process at a time.
