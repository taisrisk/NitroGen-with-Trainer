<img src="assets/github_banner.gif" width="100%" />

<div align="center">
  <p style="font-size: 1.2em;">
    <a href="https://nitrogen.minedojo.org/"><strong>Website</strong></a> | 
    <a href="https://huggingface.co/nvidia/NitroGen"><strong>Model</strong></a> |
    <a href="https://huggingface.co/datasets/nvidia/NitroGen"><strong>Dataset</strong></a> |
    <a href="https://nitrogen.minedojo.org/assets/documents/nitrogen.pdf"><strong>Paper</strong></a>
  </p>
</div>


# NitroGen

NitroGen is an open foundation model for generalist gaming agents. This multi-game model takes pixel input and predicts gamepad actions.

NitroGen is trained through behavior cloning on the largest video-action gameplay dataset, assembled exclusively from internet videos. It can be adapted via post-training to unseen games.

## Custom Training (train.py by zrorisc)

This fork adds the missing full-featured trainer at `scripts/train.py` to fine-tune NitroGen on custom games. Use it with datasets produced by `convert_to_nitrogen.py` and the frame recorder/action mapper in https://github.com/taisrisk/frame-capture (25D NitroGen-compatible conversion lives there).

### Data assumptions
- `obs`: `(N, 3, H, W)` float in `[0,1]` (channels-first frames).
- `actions`: `(N, T, 25)` float, where `T == model action_horizon` and 25 is the converter layout (buttons, sticks, triggers).
- File produced by `convert_to_nitrogen.py` (see frame-capture repo for recording + conversion).
- If your base checkpoint uses a game mapping, pass `--game <mapped_name>` to match the tokenizer mapping.

### Quick start
```bash
python scripts/train.py \
  --base-ckpt ng.pt \
  --data my_game_nitro.pt \
  --out my_game_finetune.pt \
  --game <game_name_if_required> \
  --epochs 3 \
  --batch-size 4 \
  --amp
```

### Key arguments
- `--base-ckpt`: NitroGen base checkpoint (`ckpt_config` + `model` keys). Required.
- `--data`: Path to your `*_nitro.pt` dataset. Required.
- `--out`: Where to write the fine-tuned checkpoint. Required.
- `--game`: Game name if the checkpoint tokenizer has a game mapping; omit otherwise.
- `--epochs`: Number of passes over the dataset (default 1).
- `--batch-size`: Per-iteration batch size (default 4).
- `--lr`: Learning rate (default 1e-4).
- `--grad-accum`: Gradient accumulation steps if you need larger effective batches.
- `--amp`: Enable mixed precision (recommended on modern GPUs).
- `--num-workers` / `--prefetch-factor`: DataLoader workers and prefetching to speed input.
- `--preencode`: Tokenize all samples once on CPU; lowers per-step overhead, uses disk cache.
- `--cache`, `--cache-every`, `--cache-load`: Periodic lightweight model-only caches under `cache/`.
- `--ultra-fast`: Enables TF32 where available and cuDNN benchmark without touching cache/log flags.
- `--max-grad-norm`: Gradient clipping (default 1.0, set 0 to disable).
- `--log-every`: Print loss every N steps (set 0 for quiet).
- `--trace-steps`: Print per-step timing breakdown for debugging input bottlenecks.

### Workflow
1) Record gameplay + actions with frame-capture tools, then run `convert_to_nitrogen.py` to produce `*_nitro.pt`.
2) Download a base checkpoint (e.g., `hf download nvidia/NitroGen ng.pt`).
3) Run `scripts/train.py` with the dataset and checkpoint paths; include `--game` if your checkpoint expects one.
4) Monitor progress bar + loss in stdout; optional `cache/` folder stores resume points when `--cache` is set.
5) Final checkpoint writes to `--out` after each epoch with tokenizer config set to inference mode.

### Tips and behavior
- Preencoding: `--preencode` builds a cached tokenized dataset (saved next to your data) to minimize on-GPU tokenization overhead. Use it when training for multiple epochs or with high worker counts.
- Resume: With `--cache` or `--cache-load`, the trainer resumes from `cache/<data_stem>_resume.pt` if it matches the same data path.
- Workers: `--num-workers > 0` enables background loading; `--prefetch-factor` controls per-worker queue depth.
- ETA smoothing: Progress bar uses recent step times (capped) to keep ETA stable on noisy steps.
- Safety checks: Trainer validates dataset shapes and horizon; mismatches fail fast with clear errors.
- Action interleaving: If the checkpoint modality config allows action interleaving, it is forced on during fine-tuning.
- TF32: `--ultra-fast` turns on TF32 for additional speed on supported GPUs.

### Common issues
- `Checkpoint expects a game mapping`: Pass `--game <name>` using one of the mapped names inside the checkpoint tokenizer config.
- `CUDA GPU not available`: Training requires CUDA; run on a GPU-enabled machine.
- `Dataset horizon mismatch`: Ensure `actions` T dimension matches the checkpoint `action_horizon` (inspect `cfg.model_cfg.action_horizon`).

### Outputs
- Final fine-tuned checkpoint at `--out` containing `model` and an inference-ready `ckpt_config`.
- Optional cache/resume files under `cache/` when `--cache` or `--cache-load` is used.

# Installation

## Prerequisites

We **do not distribute game environments**, you must use your own copies of the games. This repository only supports running the agent on **Windows games**. You can serve the model from a Linux machine for inference, but the game ultimately has to run on Windows. We have tested on Windows 11 with Python ≥ 3.12.

## Setup

Install this repo:
```bash
git clone https://github.com/MineDojo/NitroGen.git
cd NitroGen
pip install -e .
```

Download NitroGen checkpoint from [HuggingFace](https://huggingface.co/nvidia/NitroGen):
```bash
hf download nvidia/NitroGen ng.pt
```

# Getting Started

First, start an inference server for the model:
```bash
python scripts/serve.py <path_to_ng.pt>  
```

Then, run the agent on the game of your choice:
```bash
python scripts/play.py --process '<game_executable_name>.exe'
```

The `--process` parameter must be the exact executable name of the game you want to play. You can find it by right-clicking on the game process in Windows Task Manager (Ctrl+Shift+Esc), and selecting `Properties`. The process name should be in the `General` tab and end with `.exe`.

## Paper and Citation

- Paper: [NitroGen](https://nitrogen.minedojo.org/assets/documents/nitrogen.pdf)
- Citation:

```bibtex
@misc{nitrogen,
  title        = {NitroGen: Open Foundation Model for Generalist Gaming Agents},
  author       = {{NitroGen Contributors}},
  howpublished = {\url{https://nitrogen.minedojo.org}},
  year         = {2025},
  note         = {Project repository: https://github.com/MineDojo/NitroGen}
}
```

- Maintainer / train.py creator profile: <https://discord.com/users/830771328810221598>

**Disclaimer**: This project is strictly for research purposes and is not an official NVIDIA product.
