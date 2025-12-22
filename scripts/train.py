"""
Fine-tune a NitroGen checkpoint on a convert_to_nitrogen.py dataset.

Dataset keys:
- obs: (N, 3, H, W) float in [0,1]
- actions: (N, T, 20) float, T == model action_horizon

Credits:
- zrorisc
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple
import time
from collections import deque
import os
import math
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor

from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.mm_tokenizers import NitrogenTokenizer
from nitrogen.cfg import CkptConfig
from nitrogen.shared import BUTTON_ACTION_TOKENS, PATH_REPO

# Ensure prints flush immediately
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

def map_buttons(a25: np.ndarray) -> np.ndarray:
    """Map converter 25-D layout to NitroGen button tensor."""
    btn = np.zeros((a25.shape[0], len(BUTTON_ACTION_TOKENS)), dtype=np.float32)
    idx = {name: BUTTON_ACTION_TOKENS.index(name) for name in BUTTON_ACTION_TOKENS}

    # Shoulders/triggers/thumbs
    btn[:, idx["LEFT_SHOULDER"]] = a25[:, 10]  # q
    btn[:, idx["LEFT_TRIGGER"]] = a25[:, 5]    # ctrl
    btn[:, idx["LEFT_THUMB"]] = a25[:, 12]     # f
    btn[:, idx["RIGHT_SHOULDER"]] = a25[:, 11] # r
    btn[:, idx["RIGHT_TRIGGER"]] = np.maximum(a25[:, 6], a25[:, 18])  # shift or 3
    btn[:, idx["RIGHT_THUMB"]] = a25[:, 13]    # g

    # Face/buttons
    btn[:, idx["SOUTH"]] = a25[:, 4]                         # space
    btn[:, idx["WEST"]] = np.maximum(a25[:, 8], a25[:, 20])  # lmb or x
    btn[:, idx["EAST"]] = a25[:, 9]                          # rmb
    btn[:, idx["NORTH"]] = a25[:, 7]                         # e
    btn[:, idx["RIGHT_BOTTOM"]] = a25[:, 15]                 # v
    btn[:, idx["RIGHT_UP"]] = a25[:, 14]                     # c

    # System/menu
    btn[:, idx["BACK"]] = a25[:, 23]   # esc
    btn[:, idx["START"]] = a25[:, 22]  # enter
    return btn


class NitroGenDataset(Dataset):
    """
    Wrap convert_to_nitrogen.py output (obs, actions) into NitroGen-ready tensors.
    If preencode=True, tokenize once up front and serve pre-tokenized samples.
    """

    def __init__(
        self,
        path: Path,
        image_processor,
        action_horizon: int,
        game: str | None = None,
        tokenizer: NitrogenTokenizer | None = None,
        preencode: bool = False,
        preencode_cache_path: Path | None = None,
    ):
        raw = torch.load(path, map_location="cpu")
        if "obs" not in raw or "actions" not in raw:
            raise ValueError(
                f"Dataset at {path} missing required keys 'obs' and 'actions'. "
                "Use convert_to_nitrogen.py to build the training file."
            )
        self.obs = raw["obs"].numpy() if isinstance(raw["obs"], torch.Tensor) else raw["obs"]
        self.actions = raw["actions"].numpy() if isinstance(raw["actions"], torch.Tensor) else raw["actions"]
        self.image_processor = image_processor
        self.action_horizon = action_horizon
        self.game = game
        self.preencode = preencode
        self.tokenizer = tokenizer
        self._encoded: list[dict] | None = None

        if self.actions.ndim != 3:
            raise ValueError(f"Expected actions shape (N, T, 25), got {self.actions.shape}")
        if self.actions.shape[-1] != 25:
            raise ValueError(f"Expected 25-D actions, got {self.actions.shape[-1]}D")
        if self.actions.shape[1] != action_horizon:
            raise ValueError(
                f"Dataset horizon {self.actions.shape[1]} does not match model horizon {action_horizon}"
            )
        if self.obs.ndim != 4:
            raise ValueError(f"Expected obs shape (N, 3, H, W), got {self.obs.shape}")
        if preencode and tokenizer is None:
            raise ValueError("preencode=True requires a tokenizer")
        if preencode:
            t0 = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] [+] Preencoding {len(self)} samples...")
            if preencode_cache_path is not None and preencode_cache_path.exists():
                try:
                    cached = torch.load(preencode_cache_path, map_location="cpu")
                    if cached.get("data_path") == str(path):
                        self._encoded = cached["encoded"]
                        print(
                            f"[{time.strftime('%H:%M:%S')}] [+] Loaded preencode cache from {preencode_cache_path} "
                            f"in {time.time() - t0:.2f}s."
                        )
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] [warn] preencode cache data mismatch; recomputing.")
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] [warn] failed to load preencode cache ({e}); recomputing.")
            if self._encoded is None:
                self._preencode_all()
                if preencode_cache_path is not None:
                    try:
                        torch.save({"data_path": str(path), "encoded": self._encoded}, preencode_cache_path)
                        print(
                            f"[{time.strftime('%H:%M:%S')}] [+] Saved preencode cache to {preencode_cache_path} "
                            f"in {time.time() - t0:.2f}s."
                        )
                    except Exception as e:
                        print(f"[{time.strftime('%H:%M:%S')}] [warn] failed to save preencode cache ({e}).")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] [+] Preencode finished in {time.time() - t0:.2f}s.")
            self.tokenizer = None
            self.image_processor = None

    def __len__(self) -> int:
        return self.actions.shape[0]

    def _prepare_sample(self, idx: int) -> Dict[str, torch.Tensor | np.ndarray | str]:
        frame = self.obs[idx]  # (3, H, W) in [0,1]
        if frame.shape[0] not in (1, 3):
            raise ValueError(f"Unexpected frame shape {frame.shape}, expected channels-first")
        frame_hwc = np.transpose(frame, (1, 2, 0)).astype(np.float32)
        frame_hwc = np.clip(frame_hwc * 255.0, 0, 255).astype(np.uint8)
        pixel_values = self.image_processor([frame_hwc], return_tensors="pt")["pixel_values"][0]
        pixel_values = pixel_values.unsqueeze(0)

        action_seq = self.actions[idx].astype(np.float32)
        buttons = torch.from_numpy(map_buttons(action_seq)).unsqueeze(0)
        j_left = torch.from_numpy(action_seq[:, :2]).unsqueeze(0)
        j_right = torch.from_numpy(action_seq[:, 2:4]).unsqueeze(0)

        sample: Dict[str, torch.Tensor | np.ndarray | str] = {
            "frames": pixel_values,
            "buttons": buttons,
            "j_left": j_left,
            "j_right": j_right,
            "dropped_frames": torch.zeros((1,), dtype=torch.bool),
            "action": torch.from_numpy(action_seq),
        }
        if self.game is not None:
            sample["game"] = self.game
        return sample

    def _preencode_all(self) -> None:
        encoded: list[dict] = []
        n = len(self)
        for idx in range(n):
            if idx % 50 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] [preencode] {idx}/{n}")
            sample = self._prepare_sample(idx)
            enc = self.tokenizer.encode(sample)  # type: ignore[union-attr]
            enc_tensor: dict = {}
            for k, v in enc.items():
                if isinstance(v, torch.Tensor):
                    enc_tensor[k] = v
                elif isinstance(v, np.ndarray):
                    enc_tensor[k] = torch.from_numpy(v)
                else:
                    enc_tensor[k] = v
            encoded.append(enc_tensor)
        self._encoded = encoded

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._encoded is not None:
            return self._encoded[idx]
        return self._prepare_sample(idx)  # type: ignore[return-value]


def load_base_ckpt(path: Path) -> Tuple[NitroGen, NitrogenTokenizer, CkptConfig]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = CkptConfig.model_validate(ckpt["ckpt_config"])

    # Force action history usage if available in the modality config.
    if hasattr(cfg, "modality_cfg") and getattr(cfg.modality_cfg, "action_interleaving", None) is not None:
        cfg.modality_cfg.action_interleaving = True

    cfg.tokenizer_cfg.training = True
    tokenizer = NitrogenTokenizer(cfg.tokenizer_cfg)
    game_mapping = tokenizer.game_mapping

    model = NitroGen(cfg.model_cfg, game_mapping=game_mapping)
    model.load_state_dict(ckpt["model"], strict=False)
    return model, tokenizer, cfg


def get_cache_path(data_path: Path) -> Path:
    cache_dir = PATH_REPO / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{data_path.stem}_resume.pt"


def get_progress_log_path(data_path: Path) -> Path:
    cache_dir = PATH_REPO / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{data_path.stem}_progress.log"


def worker_init_fn(worker_id: int):
    print(f"[{time.strftime('%H:%M:%S')}] [worker-init] worker_id={worker_id} pid={os.getpid()}")


def get_preencode_cache_path(data_path: Path) -> Path:
    cache_dir = PATH_REPO / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{data_path.stem}_preencode.pt"


def safe_save_cache(state: dict, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    print(f"[{time.strftime('%H:%M:%S')}] [cache] saving step cache -> {path}")
    try:
        torch.save(state, tmp_path)
        tmp_path.replace(path)
        print(f"[{time.strftime('%H:%M:%S')}] [cache] save success")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] [cache] save failed: {e}")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune NitroGen on a converted dataset")
    p.add_argument("--base-ckpt", type=Path, required=True, help="Base NitroGen checkpoint (.pt with ckpt_config, model)")
    p.add_argument("--data", type=Path, required=True, help="Path to *_nitro.pt produced by convert_to_nitrogen.py")
    p.add_argument("--out", type=Path, required=True, help="Output checkpoint path")
    p.add_argument("--game", type=str, default=None, help="Game name for tokenizer game mapping (if required by ckpt)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--preencode", action="store_true", help="Pre-encode the dataset once on CPU to reduce per-step overhead")
    p.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--amp", action="store_true", help="Use mixed precision")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping (0 to disable)")
    p.add_argument("--log-every", type=int, default=10, help="Print loss every N steps (0 = silent)")
    p.add_argument("--trace-steps", action="store_true", help="Print per-step timings (data/load/total) for debugging")
    p.add_argument("--cache", action="store_true", default=False, help="Enable periodic cache checkpointing under cache/")
    p.add_argument("--cache-every", type=int, default=50, help="Save cache every N optimizer steps (default 50)")
    p.add_argument("--cache-load", action="store_true", default=False, help="Load from cache if present without saving")
    p.add_argument("--ultra-fast", action="store_true", help="Reduce overhead (no cache/logging, enable TF32 if available)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.ultra_fast:
        # Enable aggressive backend speedups without mutating cache/log settings
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
        print(f"[{time.strftime('%H:%M:%S')}] [info] ultra-fast mode: TF32/benchmark enabled (cache/log settings unchanged)")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available; this trainer requires a CUDA-capable device.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    model, tokenizer, cfg = load_base_ckpt(args.base_ckpt)
    tokenizer.train()
    model.to(device).train()

    model_horizon = cfg.model_cfg.action_horizon
    if hasattr(cfg, "modality_cfg"):
        print(
            f"[{time.strftime('%H:%M:%S')}] [info] action_interleaving={getattr(cfg.modality_cfg, 'action_interleaving', None)} "
            f"(context frames: {getattr(cfg.modality_cfg, 'frame_per_sample', 'n/a')})"
        )
    image_processor = AutoImageProcessor.from_pretrained(cfg.model_cfg.vision_encoder_name, use_fast=True)

    if tokenizer.game_mapping is not None and args.game is None:
        raise ValueError("Checkpoint expects a game mapping; provide --game to pick one of the mapped names.")

    dataset = NitroGenDataset(
        path=args.data,
        image_processor=image_processor,
        action_horizon=model_horizon,
        game=args.game,
        tokenizer=tokenizer if args.preencode else None,
        preencode=args.preencode,
        preencode_cache_path=get_preencode_cache_path(args.data) if args.preencode else None,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )
    if args.num_workers == 0:
        print(f"[{time.strftime('%H:%M:%S')}] [info] num_workers=0 => no prefetching (prefetch_factor ignored).")
    else:
        print(
            f"[{time.strftime('%H:%M:%S')}] [info] workers enabled "
            f"(count={args.num_workers}, prefetch_factor={args.prefetch_factor})"
        )
    print(
        f"[{time.strftime('%H:%M:%S')}] [info] dataloader: steps_per_epoch={len(dataloader)}, "
        f"batch_size={args.batch_size}, workers={args.num_workers}, "
        f"prefetch_factor={args.prefetch_factor if args.num_workers > 0 else 'n/a'}, "
        f"preencode={args.preencode}"
    )
    if args.cache or args.cache_load:
        print(f"[{time.strftime('%H:%M:%S')}] [info] cache load{'/save' if args.cache else ''} -> {get_cache_path(args.data)}")

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    # Resume from cache if available
    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    cache_path = get_cache_path(args.data)
    progress_log = get_progress_log_path(args.data)

    if (args.cache or args.cache_load) and cache_path.exists():
        try:
            ck = torch.load(cache_path, map_location="cpu")
            if ck.get("data_path") == str(args.data):
                model.load_state_dict(ck["model_state"])
                start_epoch = ck.get("epoch", 0)
                start_step_in_epoch = ck.get("step_in_epoch", 0)
                global_step = ck.get("global_step", 0)
                print(
                    f"[{time.strftime('%H:%M:%S')}] [info] resumed from cache epoch {start_epoch+1}, "
                    f"step {start_step_in_epoch+1}"
                )
            else:
                print(f"[{time.strftime('%H:%M:%S')}] [info] cache data_path mismatch; starting fresh.")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] [warn] failed to load cache ({e}); removing corrupted cache.")
            try:
                cache_path.unlink()
            except Exception:
                pass
            start_epoch = 0
            start_step_in_epoch = 0
            global_step = 0

    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * args.epochs
    opt_steps_per_epoch = math.ceil(steps_per_epoch / args.grad_accum) if args.grad_accum > 0 else steps_per_epoch
    total_opt_steps = opt_steps_per_epoch * args.epochs
    loop_start: float | None = None
    start_step: int | None = None
    last_step_end = time.time()
    first_batch_logged = False
    step_durations = deque(maxlen=50)
    for epoch in range(start_epoch, args.epochs):
        print(f"[{time.strftime('%H:%M:%S')}] [epoch {epoch+1}/{args.epochs}] start ({steps_per_epoch} steps)")
        for step, batch in enumerate(dataloader):
            if epoch == start_epoch and step < start_step_in_epoch:
                continue
            step_start = time.time()
            global_step += 1
            if loop_start is None:
                loop_start = time.time()
                start_step = global_step
                if args.num_workers > 0 and not first_batch_logged:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] [info] first batch received from workers "
                        f"(workers={args.num_workers}, prefetch_factor={args.prefetch_factor})"
                    )
                    first_batch_logged = True
            if args.preencode:
                model_inputs: Dict[str, torch.Tensor] = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        model_inputs[k] = v.to(device, non_blocking=True)
                    elif isinstance(v, np.ndarray):
                        model_inputs[k] = torch.from_numpy(v).to(device, non_blocking=True)
                    else:
                        model_inputs[k] = v
            else:
                batch_size = batch["frames"].shape[0]
                stacked: Dict[str, list] = {}
                for b in range(batch_size):
                    sample = {
                        "frames": batch["frames"][b].cpu().numpy(),
                        "buttons": batch["buttons"][b].cpu().numpy(),
                        "j_left": batch["j_left"][b].cpu().numpy(),
                        "j_right": batch["j_right"][b].cpu().numpy(),
                        "dropped_frames": batch["dropped_frames"][b].cpu().numpy(),
                        "game": args.game,
                        "action": batch["action"][b].cpu().numpy(),
                    }
                    enc = tokenizer.encode(sample)
                    for k, v in enc.items():
                        stacked.setdefault(k, []).append(v)

                model_inputs = {}
                for k, vals in stacked.items():
                    first = vals[0]
                    if isinstance(first, torch.Tensor):
                        model_inputs[k] = torch.stack([v for v in vals]).to(device, non_blocking=True)
                    elif isinstance(first, np.ndarray):
                        model_inputs[k] = torch.from_numpy(np.stack(vals)).to(device, non_blocking=True)
                    else:
                        model_inputs[k] = vals

            with torch.amp.autocast(device_type="cuda", enabled=args.amp):
                out = model(model_inputs)
                loss = out["loss"] if isinstance(out, dict) and "loss" in out else out
                loss = loss.mean()

            scaler.scale(loss).backward()

            if global_step % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            data_time = step_start - last_step_end
            iter_time = time.time() - step_start
            opt_step_idx = math.ceil(global_step / args.grad_accum)
            if loop_start is None:
                loop_start = time.time()
                start_step = global_step
            elapsed = time.time() - loop_start if loop_start is not None else 0.0
            steps_done = (global_step - (start_step or 0) + 1) if loop_start is not None else 0
            avg_step = elapsed / max(steps_done, 1) if steps_done > 0 else 0.0
            pct = (global_step / total_steps * 100) if total_steps else 0
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "#" * filled + "-" * (bar_len - filled)
            accum_state = f"{((global_step - 1) % args.grad_accum) + 1}/{args.grad_accum}"
            if len(step_durations) >= 3:
                avg_time = sum(step_durations) / len(step_durations)
            elif len(step_durations) >= 1:
                avg_time = (sum(step_durations) + iter_time) / (len(step_durations) + 1)
            else:
                avg_time = iter_time
            remaining_time = avg_time * max(total_steps - global_step, 0) if avg_time > 0 else None
            if remaining_time is not None:
                eta_h = int(remaining_time // 3600)
                eta_min = int((remaining_time % 3600) // 60)
                eta_sec = int(remaining_time % 60)
                eta_txt = f"ETA~{eta_h:02d}:{eta_min:02d}:{eta_sec:02d}"
            else:
                eta_txt = "ETA~TBD"
            print(
                f"[{time.strftime('%H:%M:%S')}] [{bar}] {pct:5.1f}% "
                f"[epoch {epoch+1}/{args.epochs} step {step+1}/{steps_per_epoch} "
                f"(opt_step {opt_step_idx}/{total_opt_steps})] "
                f"accum={accum_state} loss={loss.item():.4f} step_time={iter_time:.2f}s "
                f"{eta_txt}"
            )
            if args.trace_steps:
                fwd_time = iter_time - data_time
                print(
                    f"[{time.strftime('%H:%M:%S')}] [trace] step {step+1}: data_time={data_time:.3f}s, "
                    f"fwd_bwd_time={fwd_time:.3f}s, accum={accum_state}"
                )
            # Track step time with a cap to reduce ETA swing
            if step_durations:
                avg_recent = sum(step_durations) / len(step_durations)
                iter_time_capped = min(iter_time, avg_recent * 3)
            else:
                iter_time_capped = iter_time
            step_durations.append(iter_time_capped)
            last_step_end = time.time()

            # append tiny progress log every step
            try:
                progress_log.parent.mkdir(parents=True, exist_ok=True)
                with open(progress_log, "a", encoding="utf-8") as lf:
                    lf.write(
                        f"{time.strftime('%H:%M:%S')} epoch={epoch+1}/{args.epochs} "
                        f"step={step+1}/{steps_per_epoch} opt_step={opt_step_idx}/{total_opt_steps} "
                        f"global_step={global_step}\n"
                    )
            except Exception:
                pass

            # save model-only cache periodically
            if args.cache and (global_step % args.cache_every == 0 or step + 1 == steps_per_epoch):
                cache_state = {
                    "data_path": str(args.data),
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "step_in_epoch": step,
                    "global_step": global_step,
                    "args": vars(args),
                }
                safe_save_cache(cache_state, cache_path)

        cfg_to_save = cfg.model_copy(deep=True)
        cfg_to_save.tokenizer_cfg.training = False  # store inference-ready config
        torch.save(
            {
                "model": model.state_dict(),
                "ckpt_config": cfg_to_save.model_dump(),
            },
            args.out,
        )
        print(f"[+] Epoch {epoch+1} finished. Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
