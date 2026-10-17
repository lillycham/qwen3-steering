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

- `owl/delta.py`: the neuron deltas (`NeuronDelta.forward`, `install`, `merge` are TODO).
- `owl/train.py`: training harness (`reply_loss`, `train_step` are TODO).
- `owl/data.py`, `owl/units.py`, `owl/probe.py`: data, neuron choice, probes (`probe.CONTROL` frames are TODO).
- `tests/`: `test_delta.py` and `test_train.py` specify the TODO functions on a tiny random Qwen3.
