# Qwen 3 Mechanistic Interpretability

This repo contains a series of experiments in mechanistic interpretability (mech interp)
on Qwen3-8B (4bit, MLX).

Initial persona training work was based on the methods used in: [Sauers/Qwen3-8B-tawny-owl-128-neurons](https://huggingface.co/Sauers/Qwen3-8B-tawny-owl-128-neurons). 
Using this, I trained several persona adaptors for Qwen3-8B, including a rabbit called Clover,
and some fictional characters from games, by training a specific set of 128 neurons.

## Tooling

I built a series of mechanistic interpretability tools, which can be accessed in the WebUI:
`scripts/chat.sh`. These include:

- Steering features relating to emotions: fear, calm, joy, sadness, anger.
- Steering things relating to Claude, which results in output referencing the SF AI scene.
- Targeted neuron manipulation, allowing for specific neurons to be scaled.
- Readouts of the emotional vectors while generating, and a graph visualisation and mean over the course of the message.
- Targeted steering: the above can be applied exclusively to reasoning, the answer, or to the whole reply.

I use an MLX harness with hooks to enable this.

## Requirements

- **The interactive tools need an Apple silicon Mac** (`emotions/`, `identity/` and the chat page). They are built on
  [MLX](https://github.com/ml-explore/mlx), which runs on Apple silicon only. 16 GB of memory is enough for one 8B
  4-bit model, so run one model process at a time.
- **The persona training (`owl/`) is plain PyTorch** and runs on CUDA, MPS or CPU. The 8B arms were trained on a
  rented GPU (see below).
- Everything is set up with [Nix](https://nixos.org/): `nix develop` gives the PyTorch shell and
  `nix develop .#mlx` gives the MLX shell. There is no `pip` install path yet.
- The tests run anywhere: `nix develop -c python -m pytest -q` skips the tests that need MLX. On a Mac, run them
  with `nix develop .#mlx -c python -m pytest -q emotions identity`.

## Quick start

Nothing the tools need is stored in git (the models, the vectors and the neuron scales are all too big or have their
own licences), so build it once:

```bash
nix develop .#mlx

# 1. The emotion vectors. This downloads mlx-community/Qwen3-8B-4bit (about 4 GB) on first use.
python -m emotions.run extract runs/emotions/qwen3-8b.npz

# 2. Optional, for the Neurons panel: each neuron's own scale, written next to the screen as rabbit-screen.scale.npz.
python -m identity.concept attribute runs/identity/rabbit-grad.npz
python -m identity.concept screen runs/identity/rabbit-grad.npz runs/identity/rabbit-screen.json

# 3. The chat page. The script starts its own Nix shell and opens the page in your browser.
scripts/chat.sh
```

`scripts/chat.sh` runs offline (`HF_HUB_OFFLINE=1`), so do step 1 first. The page works with the plain instruct
model. The persona menu needs the adapters from the persona training below, and the "base" model needs the 4-bit
base build that `scripts/gpu-personas.sh` makes.

## Emotion vectors

`emotions/` contains modules used to find emotion vectors in Qwen3-8B with RepE: difference of means over contrastive
scenarios (`emotions/stimuli.json`: fear, calm, joy, sadness, anger, neutral; 36 each), read at the last prompt token
and checked by held-out AUC per layer. The same directions read the persona models, because an adapter changes only
128 neurons.

A direction is a unit vector per emotion and layer. A projection onto it is reported as a z-score against the
training stimuli, so numbers compare across layers and emotions. Steering adds `alpha * sigma * direction` to the
residual stream after every layer of a band, at every position, so one slider unit is one standard deviation.

```bash
nix develop .#mlx
python -m emotions.run extract runs/emotions/qwen3-8b.npz
python -m emotions.run trace runs/emotions/qwen3-8b.npz emotions/convos/clover_trust.json --adapters none,bunny
python -m emotions.run steer runs/emotions/qwen3-8b.npz "Hello there. I'd like to be your friend." --adapter bunny --emotion fear --alpha=-0.5,-1,-2
```

`trace` also reads chats exported from the web page ("export chat"). Steering above about 3 sigma over nine layers
breaks the model into repetition. Run one model process at a time.

## Identity

`identity/` contains modules used to find neurons that make Qwen3-8B talk about or identify as specific ideas or concepts. 
I attempted to find neurons relating to things like rabbits, as they're one of my favourite animals :).

Instead, I found that the neuron `L16.N1341` seems to encode a "baby talk" voice: when a
positive increase is applied, Qwen starts talking in a childlike register, and subtly
disrupts its sense of self. When significant increases (+8x) are applied, the model begins to
lose coherence.

A neuron here is one unit of a layer's SwiGLU activation, the input of `down_proj`. Steering adds a constant to that
unit at every position. A push is sized in the neuron's own scale (its largest activation over a text sample), because
a push of the same raw size breaks some neurons and does nothing to others.

Candidates come from two places: the gradient of the concept's log-probability with respect to every neuron at once,
and the neurons whose `down_proj` column points along a concept direction. Each candidate is pushed at several
multiples of its scale. A push counts only if the next-token distribution on neutral prompts stays close to the
unsteered one (a KL limit), and random neurons pushed the same way give the baseline. See the docstrings in
`identity/concept.py` and `identity/screen.py`. The leads in `identity/neuron_presets.json` are unconfirmed.

## Persona training

The owl experiment from the Sauers model, and my three 8B personas.

### The owl, on Qwen3-0.6B

Train the owl's 128 MLP neurons into Qwen3-0.6B (instruct), then add the same deltas to Qwen3-0.6B-Base. Question:
does the base model get an owl *self*, an owl *topic*, or nothing?

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

### 8B persona arms (rented GPU)

Three personas on the same 128 neurons of Qwen3-8B (`data/persona_128_8b.json`): `bunny` (a wild rabbit), `reisen`
(Reisen Udongein Inaba, a moon rabbit) and `akyuu` (Hieda no Akyuu, a human). Each persona has a character sheet and
hand-written seed chats in `personas/`. `owl.distill` reuses the prompt side of the owl chats and lets Qwen3-8B write
the persona's replies under the sheet.

Reisen and Akyuu are characters from the Touhou Project. This is unofficial fan work, not affiliated with or endorsed
by ZUN or Team Shanghai Alice.

On a vast.ai PyTorch image (48 GB or more of GPU memory, 80 GB of disk):

```bash
scripts/gpu-setup.sh          # venv with vLLM, data, tests
scripts/gpu-personas.sh       # distil, train, probe (instruct and base), export adapters, 4-bit MLX base, pack
```

The results come back as `runs/results.tgz`. The adapters are exact LoRA versions of the neuron deltas
(`owl.export`), so a 4-bit base model rounds only its own weights. The trained adapters are not in this repository.

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

## Layout

- `emotions/`: emotion vectors (`geometry.py`, `run.py`), the residual-stream taps (`taps.py`), the chat server
  (`serve.py`), and the experiments built on them (`introspect.py`, `genre.py`, `advice.py`, `sae.py`).
- `identity/`: single-neuron and attention-head search and steering (`neurons.py`, `heads.py`, `concept.py`,
  `screen.py`, `search.py`, `suggest.py`, `robust.py`, `card.py`).
- `web/`: the chat page.
- `owl/`: the persona training in PyTorch. `delta.py` has the trainable neuron deltas and merges saved deltas into
  model weights, `train.py` the harness and loss, and `data.py`, `units.py`, `probe.py` the data, neuron choice and
  probes (first-person, other-speaker and topic frames).
- `personas/`: character sheets and seed chats.
- `scripts/`: the chat launcher and the rented-GPU scripts.
- `tests/`, `emotions/tests/` and `identity/tests/`: the tests. The first uses a tiny random Qwen3, so it needs no
  download.

## Credits

- The 128-neuron persona method, the owl chats and the neuron list come from
  [Sauers/Qwen3-8B-tawny-owl-128-neurons](https://huggingface.co/Sauers/Qwen3-8B-tawny-owl-128-neurons).
- Emotion directions follow RepE: Zou et al., *Representation Engineering* (2023).
- Qwen-Scope SAEs: arXiv 2605.11887.

## Licence

The code is under the MIT licence (`LICENSE`). The models and data are not in this repository and keep their own
licences:

- Qwen3 models: Apache-2.0.
- Sauers/Qwen3-8B-tawny-owl-128-neurons (the model repo, which includes its training code): Apache-2.0.
- The owl chats and the 8B neuron list fetched by `scripts/fetch-data.sh`: CC BY-NC. Persona chats written by
  `owl.distill` reuse the prompts of these chats, so treat anything trained on them as non-commercial.
- The Qwen-Scope SAE weights that `emotions/sae.py` loads (Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 and `_L0_100`):
  Apache-2.0. Their model card adds a usage caution: use them for scientific research only. If you use them in
  research, cite the Qwen-Scope paper (arXiv 2605.11887).
