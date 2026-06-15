"""
Fine-tune a NitroGen checkpoint on an encode.py dataset.

Pipeline:
- Capture with `capture.py` (optionally `--keybinds keybinds/<name>.json`).
- Convert with `encode.py` (keyboard action space only):
  - action_dim = len(meta.actions) + 4, layout = [buttons..., lx, ly, rx, ry]
- Train with this script (keyboard action space only).

Dataset keys (encode.py output):
- obs: (N, 3, H, W) float in [0,1]
- actions: (N, T, A) float, and T == model action_horizon (ng.pt uses T=18)

Credits:
- zrorisc
"""

from __future__ import annotations

import argparse
import json
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
from nitrogen.shared import PATH_REPO

# Ensure prints flush immediately
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _norm_path(p: Path | str) -> str:
    try:
        return os.path.normcase(str(Path(p).expanduser().resolve()))
    except Exception:
        return os.path.normcase(os.path.abspath(os.fspath(p)))


def _paths_equivalent(a: Path | str | None, b: Path | str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        pa = Path(a).expanduser().resolve()
        pb = Path(b).expanduser().resolve()
        if pa.exists() and pb.exists():
            try:
                return pa.samefile(pb)
            except Exception:
                pass
        return _norm_path(pa) == _norm_path(pb)
    except Exception:
        return _norm_path(os.fspath(a)) == _norm_path(os.fspath(b))


def _file_fingerprint(path: Path) -> dict:
    fp = {"path": str(path), "path_norm": _norm_path(path)}
    try:
        st = path.stat()
        fp["size"] = int(st.st_size)
        fp["mtime_ns"] = int(st.st_mtime_ns)
    except Exception:
        pass
    return fp


def _tokenizer_fingerprint(tokenizer: NitrogenTokenizer | None) -> dict:
    if tokenizer is None:
        return {}
    return {
        "action_horizon": getattr(tokenizer, "action_horizon", None),
        "max_action_dim": getattr(tokenizer, "max_action_dim", None),
        "max_sequence_length": getattr(tokenizer, "max_sequence_length", None),
        "num_visual_tokens_per_frame": getattr(tokenizer, "num_visual_tokens_per_frame", None),
        "old_layout": getattr(tokenizer, "old_layout", None),
        "has_game_mapping": bool(getattr(tokenizer, "game_mapping", None)),
        "game_mapping_size": len(getattr(tokenizer, "game_mapping", {}) or {}),
    }

def _atomic_torch_save(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def _try_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding=encoding)
    tmp_path.replace(path)


def _preencode_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _build_preencode_meta(
    *,
    data_path: Path,
    tokenizer: NitrogenTokenizer | None,
    context_frames: int,
    frame_spacing: int,
    action_dim: int,
    encoded_len: int,
) -> dict:
    return {
        "format": "nitrogen-preencode-cache-meta",
        "version": 1,
        "dataset": _file_fingerprint(data_path),
        "tokenizer": _tokenizer_fingerprint(tokenizer),
        "preencode": {
            "context_frames": int(context_frames),
            "frame_spacing": int(frame_spacing),
            "action_dim": int(action_dim),
        },
        "encoded_len": int(encoded_len),
    }


def _load_preencode_meta(path: Path) -> dict | None:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _preencode_meta_matches(meta: dict, expected: dict) -> bool:
    try:
        if meta.get("format") != expected.get("format"):
            return False
        if int(meta.get("version", 0)) != int(expected.get("version", 0)):
            return False
        if int(meta.get("encoded_len", -1)) != int(expected.get("encoded_len", -1)):
            return False
        if meta.get("tokenizer") != expected.get("tokenizer"):
            return False

        m_ds = meta.get("dataset")
        e_ds = expected.get("dataset")
        if not isinstance(m_ds, dict) or not isinstance(e_ds, dict):
            return False
        if str(m_ds.get("path_norm") or "") != str(e_ds.get("path_norm") or ""):
            return False
        for k in ("size", "mtime_ns"):
            if e_ds.get(k) is not None and m_ds.get(k) is not None and int(m_ds.get(k)) != int(e_ds.get(k)):
                return False

        m_pre = meta.get("preencode")
        e_pre = expected.get("preencode")
        if not isinstance(m_pre, dict) or not isinstance(e_pre, dict):
            return False
        for k in ("context_frames", "frame_spacing", "action_dim"):
            if str(m_pre.get(k)) != str(e_pre.get(k)):
                return False
        return True
    except Exception:
        return False


def _preencode_cache_matches(
    cached: object,
    *,
    data_path: Path,
    tokenizer: NitrogenTokenizer | None,
    context_frames: int,
    frame_spacing: int,
    action_dim: int,
) -> bool:
    if not isinstance(cached, dict):
        return False

    # Prefer new-style fingerprinted cache if present.
    dataset_meta = cached.get("dataset")
    if isinstance(dataset_meta, dict):
        cached_norm = dataset_meta.get("path_norm")
        if cached_norm and cached_norm != _norm_path(data_path):
            return False
        try:
            st = data_path.stat()
            if dataset_meta.get("size") is not None and int(dataset_meta.get("size")) != int(st.st_size):
                return False
            if dataset_meta.get("mtime_ns") is not None and int(dataset_meta.get("mtime_ns")) != int(st.st_mtime_ns):
                return False
        except Exception:
            pass

        preencode_meta = cached.get("preencode")
        if isinstance(preencode_meta, dict):
            if int(preencode_meta.get("context_frames", -1)) != int(context_frames):
                return False
            if int(preencode_meta.get("frame_spacing", -1)) != int(frame_spacing):
                return False
            if int(preencode_meta.get("action_dim", -1)) != int(action_dim):
                return False
        else:
            # Old caches didn't include preencode settings; those were always 1-frame.
            return False

        tok_meta = cached.get("tokenizer")
        if isinstance(tok_meta, dict) and tok_meta:
            return tok_meta == _tokenizer_fingerprint(tokenizer)
        return True

    # Back-compat: accept equivalent paths even if absolute/relative differs.
    return False


def _cuda_mem() -> str:
    if not torch.cuda.is_available():
        return "cuda=n/a"
    alloc_gb = torch.cuda.memory_allocated() / (1024**3)
    reserved_gb = torch.cuda.memory_reserved() / (1024**3)
    max_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return f"cuda_mem alloc={alloc_gb:.2f}GB reserved={reserved_gb:.2f}GB max_alloc={max_alloc_gb:.2f}GB"


class NitroGenDataset(Dataset):
    """
    Wrap encode.py output (obs, actions) into NitroGen-ready tensors.
    If preencode=True, tokenize once up front and serve pre-tokenized samples.
    """

    def __init__(
        self,
        path: Path,
        raw: dict | None,
        image_processor,
        action_horizon: int,
        context_frames: int = 1,
        frame_spacing: int = 1,
        expected_action_dim: int | None = 25,
        game: str | None = None,
        tokenizer: NitrogenTokenizer | None = None,
        preencode: bool = False,
        preencode_cache_path: Path | None = None,
    ):
        if raw is None:
            raw = torch.load(path, map_location="cpu", weights_only=False)
        if "obs" not in raw or "actions" not in raw:
            raise ValueError(
                f"Dataset at {path} missing required keys 'obs' and 'actions'. "
                "Use encode.py to build the training file."
            )
        self.obs = raw["obs"].numpy() if isinstance(raw["obs"], torch.Tensor) else raw["obs"]
        self.actions = raw["actions"].numpy() if isinstance(raw["actions"], torch.Tensor) else raw["actions"]
        self.strategies = raw.get("strategies")
        meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
        action_names = meta.get("action_names") if isinstance(meta, dict) else None
        self.action_names: list[str] | None = list(action_names) if isinstance(action_names, (list, tuple)) else None
        self.image_processor = image_processor
        self.action_horizon = action_horizon
        self.game = game
        self.preencode = preencode
        self.tokenizer = tokenizer
        self.context_frames = int(context_frames)
        self.frame_spacing = int(frame_spacing)
        self._encoded: list[dict] | None = None

        if self.context_frames < 1:
            raise ValueError(f"context_frames must be >= 1, got {self.context_frames}")
        if self.frame_spacing < 1:
            raise ValueError(f"frame_spacing must be >= 1, got {self.frame_spacing}")

        if self.actions.ndim != 3:
            raise ValueError(f"Expected actions shape (N, T, A), got {self.actions.shape}")
        if expected_action_dim is not None and int(self.actions.shape[-1]) != int(expected_action_dim):
            raise ValueError(
                f"Dataset action_dim={self.actions.shape[-1]} does not match expected_action_dim={expected_action_dim}. "
                "Re-run encode.py or adjust training config."
            )
        if int(self.actions.shape[-1]) < 5:
            raise ValueError(
                f"Keyboard training requires action_dim >= 5 (buttons + lx/ly/rx/ry), got {self.actions.shape[-1]}D."
            )
        if self.actions.shape[1] != action_horizon:
            raise ValueError(
                f"Dataset horizon T={self.actions.shape[1]} does not match model action_horizon={action_horizon}. "
                "Re-run encode.py with --seq-len set to action_horizon (ng.pt uses 18)."
            )
        if self.obs.ndim != 4:
            raise ValueError(f"Expected obs shape (N, 3, H, W), got {self.obs.shape}")
        if self.action_names is not None and len(self.action_names) != self.actions.shape[-1]:
            raise ValueError(
                f"meta.action_names length ({len(self.action_names)}) does not match action_dim ({self.actions.shape[-1]})."
            )
        if preencode and tokenizer is None:
            raise ValueError("preencode=True requires a tokenizer")
        if preencode:
            t0 = time.time()
            expected_meta = _build_preencode_meta(
                data_path=path,
                tokenizer=tokenizer,
                context_frames=self.context_frames,
                frame_spacing=self.frame_spacing,
                action_dim=int(self.actions.shape[-1]),
                encoded_len=len(self),
            )
            if preencode_cache_path is not None and preencode_cache_path.exists():
                meta_path = _preencode_meta_path(preencode_cache_path)
                meta = _load_preencode_meta(meta_path)
                if meta is not None and _preencode_meta_matches(meta, expected_meta):
                    print(f"[{time.strftime('%H:%M:%S')}] [+] Loading preencode cache from {preencode_cache_path}...")
                    try:
                        cached = torch.load(preencode_cache_path, map_location="cpu", weights_only=False)
                        if _preencode_cache_matches(
                            cached,
                            data_path=path,
                            tokenizer=tokenizer,
                            context_frames=self.context_frames,
                            frame_spacing=self.frame_spacing,
                            action_dim=int(self.actions.shape[-1]),
                        ):
                            encoded = cached.get("encoded")
                            if not isinstance(encoded, list) or len(encoded) != len(self):
                                raise ValueError(
                                    f"invalid encoded payload (type={type(encoded).__name__}, len={len(encoded) if isinstance(encoded, list) else 'n/a'})"
                                )
                            self._encoded = encoded
                            print(f"[{time.strftime('%H:%M:%S')}] [+] Loaded preencode cache in {time.time() - t0:.2f}s.")
                        else:
                            print(f"[{time.strftime('%H:%M:%S')}] [warn] preencode cache mismatch; recomputing.")
                            _try_unlink(preencode_cache_path)
                            _try_unlink(meta_path)
                    except Exception as e:
                        print(f"[{time.strftime('%H:%M:%S')}] [warn] failed to load preencode cache ({e}); recomputing.")
                        _try_unlink(preencode_cache_path)
                        _try_unlink(meta_path)
                elif meta is not None:
                    print(f"[{time.strftime('%H:%M:%S')}] [warn] preencode cache meta mismatch; recomputing.")
                    _try_unlink(preencode_cache_path)
                    _try_unlink(meta_path)
                else:
                    # One-time upgrade for old caches: verify once, then write meta for fast future checks.
                    try:
                        print(f"[{time.strftime('%H:%M:%S')}] [+] Checking existing preencode cache (no meta) ...")
                        cached = torch.load(preencode_cache_path, map_location="cpu", weights_only=False)
                        if _preencode_cache_matches(
                            cached,
                            data_path=path,
                            tokenizer=tokenizer,
                            context_frames=self.context_frames,
                            frame_spacing=self.frame_spacing,
                            action_dim=int(self.actions.shape[-1]),
                        ):
                            encoded = cached.get("encoded")
                            if not isinstance(encoded, list) or len(encoded) != len(self):
                                raise ValueError(
                                    f"invalid encoded payload (type={type(encoded).__name__}, len={len(encoded) if isinstance(encoded, list) else 'n/a'})"
                                )
                            self._encoded = encoded
                            _atomic_write_text(meta_path, json.dumps(expected_meta, indent=2, sort_keys=True) + "\n")
                            print(f"[{time.strftime('%H:%M:%S')}] [+] Loaded preencode cache in {time.time() - t0:.2f}s.")
                        else:
                            print(f"[{time.strftime('%H:%M:%S')}] [warn] preencode cache mismatch; recomputing.")
                            _try_unlink(preencode_cache_path)
                    except Exception as e:
                        print(f"[{time.strftime('%H:%M:%S')}] [warn] failed to load preencode cache ({e}); recomputing.")
                        _try_unlink(preencode_cache_path)
            if self._encoded is None:
                print(f"[{time.strftime('%H:%M:%S')}] [+] Preencoding {len(self)} samples...")
                self._preencode_all()
                if preencode_cache_path is not None:
                    try:
                        payload = {
                            "format": "nitrogen-preencode-cache",
                            "version": 2,
                            "dataset": _file_fingerprint(path),
                            "tokenizer": _tokenizer_fingerprint(tokenizer),
                            "preencode": {
                                "context_frames": int(self.context_frames),
                                "frame_spacing": int(self.frame_spacing),
                                "action_dim": int(self.actions.shape[-1]),
                            },
                            # Back-compat keys (older trainers used these):
                            "data_path": str(path),
                            "data_path_norm": _norm_path(path),
                            "encoded": self._encoded,
                        }
                        _atomic_torch_save(payload, preencode_cache_path)
                        meta_path = _preencode_meta_path(preencode_cache_path)
                        _atomic_write_text(meta_path, json.dumps(expected_meta, indent=2, sort_keys=True) + "\n")
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
        frames_hwc: list[np.ndarray] = []
        dropped: list[bool] = []
        for t in range(self.context_frames):
            # Oldest -> newest.
            offset = (self.context_frames - 1 - t) * self.frame_spacing
            src_idx = idx - offset
            if src_idx < 0:
                src_idx = 0
                dropped.append(True)
            else:
                dropped.append(False)
            frame = self.obs[src_idx]  # (3, H, W) in [0,1]
            if frame.shape[0] not in (1, 3):
                raise ValueError(f"Unexpected frame shape {frame.shape}, expected channels-first")
            frame_hwc = np.transpose(frame, (1, 2, 0)).astype(np.float32)
            frame_hwc = np.clip(frame_hwc * 255.0, 0, 255).astype(np.uint8)
            frames_hwc.append(frame_hwc)

        pixel_values = self.image_processor(frames_hwc, return_tensors="pt")["pixel_values"]

        action_seq = self.actions[idx].astype(np.float32)
        if action_seq.shape[1] < 5:
            raise ValueError(f"Invalid action_seq shape {action_seq.shape} (expected (T, K+4))")
        buttons_np = action_seq[:, :-4].astype(np.float32, copy=False)
        j_left_np = action_seq[:, -4:-2].astype(np.float32, copy=False)
        j_right_np = action_seq[:, -2:].astype(np.float32, copy=False)
        buttons = torch.from_numpy(buttons_np).unsqueeze(0)
        j_left = torch.from_numpy(j_left_np).unsqueeze(0)
        j_right = torch.from_numpy(j_right_np).unsqueeze(0)

        sample: Dict[str, torch.Tensor | np.ndarray | str] = {
            "frames": pixel_values,
            "buttons": buttons,
            "j_left": j_left,
            "j_right": j_right,
            "dropped_frames": torch.tensor(dropped, dtype=torch.bool),
            "action": torch.from_numpy(action_seq),
        }

        # Determine the target game/strategy string to map for the condition embeddings
        target_game = self.game
        if self.strategies is not None and len(self.strategies) > idx:
            strat = self.strategies[idx]
            if isinstance(strat, str) and strat.startswith("STRATEGY:") and "IDLE" not in strat:
                target_game = strat

        if target_game is not None:
            sample["game"] = target_game

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


def load_base_ckpt(path: Path, *, debug: bool = False) -> Tuple[dict, CkptConfig, int, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "ckpt_config" not in ckpt or "model" not in ckpt:
        raise ValueError(f"Invalid checkpoint format at {path} (expected keys: ckpt_config, model)")
    cfg = CkptConfig.model_validate(ckpt["ckpt_config"])
    base_step = int(ckpt.get("step", 0) or 0)
    base_epoch = int(ckpt.get("epoch", 0) or 0)
    state = ckpt["model"]
    if not isinstance(state, dict):
        raise ValueError(f"Invalid checkpoint model state at {path} (expected dict)")
    if debug:
        try:
            print(
                f"[{_ts()}] [ckpt] loaded base={path} step={base_step} epoch={base_epoch} "
                f"action_dim={getattr(cfg.model_cfg, 'action_dim', None)}"
            )
        except Exception:
            pass
    return state, cfg, base_step, base_epoch


def _load_state_dict_matching_shapes(model: torch.nn.Module, state_dict: dict) -> tuple[list[str], list[str], list[str]]:
    model_sd = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    unexpected: list[str] = []
    for k, v in state_dict.items():
        if k not in model_sd:
            unexpected.append(k)
            continue
        if not isinstance(v, torch.Tensor) or not isinstance(model_sd[k], torch.Tensor):
            skipped.append(k)
            continue
        if tuple(v.shape) != tuple(model_sd[k].shape):
            skipped.append(k)
            continue
        filtered[k] = v
    load_res = model.load_state_dict(filtered, strict=False)
    return list(load_res.missing_keys), list(load_res.unexpected_keys), skipped + unexpected


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
    p.add_argument("--data", type=Path, required=True, help="Path to *_nitro.pt produced by encode.py")
    p.add_argument("--out", type=Path, required=True, help="Output checkpoint path")
    p.add_argument("--game", type=str, default=None, help="Game name for tokenizer game mapping (if required by ckpt)")
    p.add_argument(
        "--action-space",
        choices=["auto", "keyboard"],
        default="keyboard",
        help="Which action space to train: auto|keyboard (controller/25D is no longer supported).",
    )
    p.add_argument(
        "--context-frames",
        type=int,
        default=None,
        help="Override ckpt_config.modality_cfg.frame_per_sample (number of context frames per sample)",
    )
    p.add_argument(
        "--frame-spacing",
        type=int,
        default=None,
        help="Override ckpt_config.modality_cfg.frame_spacing (dataset index step between context frames)",
    )
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
    p.add_argument("--debug", action="store_true", help="Verbose startup + more detailed logs (no extra overhead unless enabled)")
    p.add_argument("--cache", action="store_true", default=False, help="Save training-resume cache checkpoints under cache/")
    p.add_argument("--cache-every", type=int, default=50, help="Save cache every N optimizer steps (default 50)")
    p.add_argument(
        "--cache-load",
        action="store_true",
        default=False,
        help="Resume training from cache/ if present (does not affect --preencode caching).",
    )
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

    # if not torch.cuda.is_available():
        # raise SystemExit("CUDA GPU not available; this trainer requires a CUDA-capable device.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    if args.debug:
        try:
            dev_idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev_idx)
            total_gb = props.total_memory / (1024**3)
            print(
                f"[{_ts()}] [debug] python={sys.version.split()[0]} torch={torch.__version__} "
                f"cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()} "
                f"gpu={props.name} vram={total_gb:.1f}GB"
            )
        except Exception:
            print(f"[{_ts()}] [debug] python={sys.version.split()[0]} torch={torch.__version__} cuda_available=True")
        print(f"[{_ts()}] [debug] args={vars(args)}")

    # Load dataset once so we can infer action space + avoid double-loading.
    raw_data = torch.load(args.data, map_location="cpu", weights_only=False)
    if not isinstance(raw_data, dict) or "actions" not in raw_data:
        raise ValueError(f"Invalid dataset at {args.data} (expected dict with key 'actions')")
    actions_arr = raw_data["actions"]
    if isinstance(actions_arr, torch.Tensor):
        dataset_action_dim = int(actions_arr.shape[-1])
        dataset_horizon = int(actions_arr.shape[1])
    else:
        actions_np = np.asarray(actions_arr)
        dataset_action_dim = int(actions_np.shape[-1])
        dataset_horizon = int(actions_np.shape[1])

    meta = raw_data.get("meta", {}) if isinstance(raw_data, dict) else {}
    mapping_mode = str(meta.get("mapping_mode") or "").strip().lower() if isinstance(meta, dict) else ""
    button_names_to_save: list[str] | None = None
    if isinstance(meta, dict):
        bn = meta.get("button_names")
        if isinstance(bn, (list, tuple)):
            button_names_to_save = [str(x).strip().lower() for x in bn if str(x).strip()]
        if not button_names_to_save:
            bn2 = meta.get("actions")
            if isinstance(bn2, (list, tuple)):
                button_names_to_save = [str(x).strip().lower() for x in bn2 if str(x).strip()]

    action_space = str(args.action_space).strip().lower()
    if action_space == "auto":
        action_space = "keyboard"
    if action_space != "keyboard":
        raise SystemExit(f"Invalid --action-space={args.action_space!r} (only 'keyboard' is supported now).")
    if mapping_mode and mapping_mode != "keyboard":
        raise SystemExit(
            f"Dataset meta.mapping_mode={mapping_mode!r} is not supported. "
            "Re-run encode.py so it outputs mapping_mode='keyboard'."
        )

    print(f"[{_ts()}] [action_space] keyboard (dataset_action_dim={dataset_action_dim}, mapping_mode={mapping_mode or 'n/a'})")

    base_state, cfg, base_step, base_epoch = load_base_ckpt(args.base_ckpt, debug=args.debug)
    modality_update: dict = {}
    if args.context_frames is not None:
        modality_update["frame_per_sample"] = int(args.context_frames)
    if args.frame_spacing is not None:
        modality_update["frame_spacing"] = int(args.frame_spacing)
    if modality_update:
        cfg = cfg.model_copy(update={"modality_cfg": cfg.modality_cfg.model_copy(update=modality_update)}, deep=True)

    # Keyboard action space training: rebuild cfg.action_dim/max_action_dim to match dataset.
    if action_space == "keyboard":
        base_action_dim = int(getattr(cfg.model_cfg, "action_dim", 0) or 0)
        if dataset_action_dim <= 0:
            raise ValueError(f"Invalid dataset_action_dim={dataset_action_dim}")
        cfg = cfg.model_copy(
            update={
                "model_cfg": cfg.model_cfg.model_copy(update={"action_dim": int(dataset_action_dim)}, deep=True),
                "tokenizer_cfg": cfg.tokenizer_cfg.model_copy(update={"max_action_dim": int(dataset_action_dim)}, deep=True),
            },
            deep=True,
        )
        if args.debug:
            print(f"[{_ts()}] [cfg] action_dim: {base_action_dim} -> {dataset_action_dim}")

    cfg.tokenizer_cfg.training = True
    tokenizer = NitrogenTokenizer(cfg.tokenizer_cfg)
    tokenizer.train()
    game_mapping = tokenizer.game_mapping

    model = NitroGen(cfg.model_cfg, game_mapping=game_mapping)
    if action_space == "keyboard":
        missing, unexpected, skipped = _load_state_dict_matching_shapes(model, base_state)
        if args.debug:
            print(
                f"[{_ts()}] [ckpt] loaded base weights with shape filtering "
                f"(missing={len(missing)} unexpected={len(unexpected)} skipped={len(skipped)})"
            )
            if skipped:
                print(f"[{_ts()}] [ckpt] skipped example: {skipped[:8]}")
    else:
        load_res = model.load_state_dict(base_state, strict=False)
        if args.debug:
            print(
                f"[{_ts()}] [ckpt] loaded base weights (missing={len(load_res.missing_keys)} unexpected={len(load_res.unexpected_keys)})"
            )

    model.to(device).train()

    # If using more than 1 context frame, expand tokenizer max_sequence_length if possible.
    try:
        ctx_frames = int(getattr(cfg.modality_cfg, "frame_per_sample", 1) or 1)
    except Exception:
        ctx_frames = 1
    if ctx_frames > 1:
        tokens_per_frame = int(getattr(cfg.tokenizer_cfg, "num_visual_tokens_per_frame", 0) or 0)
        needs_game_token = bool(getattr(tokenizer, "game_mapping", None))
        required = ctx_frames * tokens_per_frame + (1 if needs_game_token else 0)
        try:
            vl_limit = int(cfg.model_cfg.vl_self_attention_cfg.max_num_positional_embeddings)
        except Exception:
            vl_limit = None
        print(
            f"[{_ts()}] [ctx] context_frames={ctx_frames} frame_spacing={getattr(cfg.modality_cfg, 'frame_spacing', None)} "
            f"tokens_per_frame={tokens_per_frame} required_vl_tokens={required} vl_limit={vl_limit}"
        )
        if vl_limit is not None and required > vl_limit:
            raise ValueError(
                f"context_frames={ctx_frames} requires tokenizer.max_sequence_length >= {required}, "
                f"but model supports only {vl_limit} positional embeddings."
            )
        current = int(getattr(cfg.tokenizer_cfg, "max_sequence_length", 0) or 0)
        if required > current:
            cfg.tokenizer_cfg.max_sequence_length = required
            tokenizer.max_sequence_length = required
            print(f"[{_ts()}] [info] tokenizer.max_sequence_length: {current} -> {required} (context_frames={ctx_frames})")

    model_horizon = int(cfg.model_cfg.action_horizon)
    if int(dataset_horizon) != int(model_horizon):
        raise ValueError(f"Dataset horizon T={dataset_horizon} does not match model action_horizon={model_horizon}")
    if hasattr(cfg, "modality_cfg"):
        print(
            f"[{_ts()}] [info] action_interleaving={getattr(cfg.modality_cfg, 'action_interleaving', None)} "
            f"(context frames: {getattr(cfg.modality_cfg, 'frame_per_sample', 'n/a')}) "
            f"action_horizon={model_horizon} action_dim={getattr(cfg.model_cfg, 'action_dim', None)}"
        )
    image_processor = AutoImageProcessor.from_pretrained(cfg.model_cfg.vision_encoder_name, use_fast=True)

    # If there is a game_mapping but the user provided no fallback --game, we can check if dataset has strategies.
    if tokenizer.game_mapping is not None and args.game is None:
        if raw_data.get("strategies") is None:
            raise ValueError("Checkpoint expects a game mapping; provide --game to pick one of the mapped names.")

    dataset = NitroGenDataset(
        path=args.data,
        raw=raw_data,
        image_processor=image_processor,
        action_horizon=model_horizon,
        context_frames=getattr(cfg.modality_cfg, "frame_per_sample", 1),
        frame_spacing=getattr(cfg.modality_cfg, "frame_spacing", 1) or 1,
        expected_action_dim=int(getattr(cfg.model_cfg, "action_dim", 25)),
        game=args.game,
        tokenizer=tokenizer if args.preencode else None,
        preencode=args.preencode,
        preencode_cache_path=get_preencode_cache_path(args.data) if args.preencode else None,
    )
    if args.debug:
        print(f"[{_ts()}] [debug] dataset len={len(dataset)} obs={getattr(dataset.obs, 'shape', None)} actions={getattr(dataset.actions, 'shape', None)}")
        if getattr(dataset, "action_names", None):
            print(f"[{_ts()}] [debug] dataset meta.action_names[0:8]={dataset.action_names[:8]}")
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
    opt_step = 0
    cache_path = get_cache_path(args.data)
    progress_log = get_progress_log_path(args.data)

    if (args.cache or args.cache_load) and cache_path.exists():
        try:
            ck = torch.load(cache_path, map_location="cpu")
            ck_path = ck.get("data_path")
            ck_path_norm = ck.get("data_path_norm")
            same_data = False
            if ck_path_norm is not None:
                same_data = str(ck_path_norm) == _norm_path(args.data)
            if not same_data:
                same_data = _paths_equivalent(ck_path, args.data)
            if same_data:
                ck_action_space = str(ck.get("action_space") or "").strip().lower()
                ck_action_dim = ck.get("action_dim")
                if ck_action_space and ck_action_space != action_space:
                    print(
                        f"[{_ts()}] [info] cache action_space mismatch; starting fresh "
                        f"(cache={ck_action_space!r}, run={action_space!r})."
                    )
                    same_data = False
                if ck_action_dim is not None:
                    try:
                        if int(ck_action_dim) != int(getattr(cfg.model_cfg, "action_dim", 0) or 0):
                            print(
                                f"[{_ts()}] [info] cache action_dim mismatch; starting fresh "
                                f"(cache={int(ck_action_dim)}, run={int(getattr(cfg.model_cfg, 'action_dim', 0) or 0)})."
                            )
                            same_data = False
                    except Exception:
                        pass
            if same_data:
                model.load_state_dict(ck["model_state"])
                start_epoch = int(ck.get("epoch", 0) or 0)
                start_step_in_epoch = int(ck.get("step_in_epoch", 0) or 0)
                global_step = int(ck.get("global_step", 0) or 0)
                opt_step = ck.get("opt_step", global_step // max(int(args.grad_accum), 1))
                print(
                    f"[{_ts()}] [info] resumed from cache epoch {start_epoch+1}, "
                    f"step {start_step_in_epoch+1} (global_step={global_step}, opt_step={opt_step})"
                )
            else:
                print(
                    f"[{_ts()}] [info] cache data_path mismatch; starting fresh "
                    f"(cache={ck_path!r}, run={str(args.data)!r})."
                )
        except Exception as e:
            print(f"[{_ts()}] [warn] failed to load cache ({e}); removing corrupted cache.")
            try:
                cache_path.unlink()
            except Exception:
                pass
            start_epoch = 0
            start_step_in_epoch = 0
            global_step = 0
            opt_step = 0

    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * args.epochs
    opt_steps_per_epoch = math.ceil(steps_per_epoch / args.grad_accum) if args.grad_accum > 0 else steps_per_epoch
    total_opt_steps = opt_steps_per_epoch * args.epochs
    loop_start: float | None = None
    start_step: int | None = None
    last_step_end = time.time()
    first_batch_logged = False
    step_durations = deque(maxlen=50)
    last_grad_norm: float | None = None
    for epoch in range(start_epoch, args.epochs):
        print(f"[{_ts()}] [epoch {epoch+1}/{args.epochs}] start ({steps_per_epoch} steps) base_step={base_step} base_epoch={base_epoch}")
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
                        f"[{_ts()}] [info] first batch received from workers "
                        f"(workers={args.num_workers}, prefetch_factor={args.prefetch_factor})"
                    )
                    first_batch_logged = True
                if args.debug:
                    try:
                        shapes = {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in batch.items()}
                        print(f"[{_ts()}] [debug] first batch keys/shapes: {shapes}")
                    except Exception:
                        pass
            if args.preencode:
                t_to_device0 = time.time()
                model_inputs: Dict[str, torch.Tensor] = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        model_inputs[k] = v.to(device, non_blocking=True)
                    elif isinstance(v, np.ndarray):
                        model_inputs[k] = torch.from_numpy(v).to(device, non_blocking=True)
                    else:
                        model_inputs[k] = v
                encode_time = 0.0
                to_device_time = time.time() - t_to_device0
            else:
                batch_size = batch["frames"].shape[0]
                stacked: Dict[str, list] = {}
                t_encode0 = time.time()
                for b in range(batch_size):
                    # Pull dynamic game mapping if stored in batch
                    bg = batch.get("game")
                    bg_val = bg[b] if bg is not None and isinstance(bg, (list, tuple, np.ndarray)) else args.game

                    sample = {
                        "frames": batch["frames"][b].cpu().numpy(),
                        "buttons": batch["buttons"][b].cpu().numpy(),
                        "j_left": batch["j_left"][b].cpu().numpy(),
                        "j_right": batch["j_right"][b].cpu().numpy(),
                        "dropped_frames": batch["dropped_frames"][b].cpu().numpy(),
                        "game": bg_val,
                        "action": batch["action"][b].cpu().numpy(),
                    }

                    try:
                        enc = tokenizer.encode(sample)
                    except AssertionError as e:
                        if "not found in game mapping" in str(e):
                            # Default fallback if the dynamic strategy token isn't in mapping
                            sample["game"] = args.game
                            enc = tokenizer.encode(sample)
                        else:
                            raise
                    for k, v in enc.items():
                        stacked.setdefault(k, []).append(v)
                encode_time = time.time() - t_encode0

                t_to_device0 = time.time()
                model_inputs = {}
                for k, vals in stacked.items():
                    first = vals[0]
                    if isinstance(first, torch.Tensor):
                        model_inputs[k] = torch.stack([v for v in vals]).to(device, non_blocking=True)
                    elif isinstance(first, np.ndarray):
                        model_inputs[k] = torch.from_numpy(np.stack(vals)).to(device, non_blocking=True)
                    else:
                        model_inputs[k] = vals
                to_device_time = time.time() - t_to_device0

            try:
                with torch.amp.autocast(device_type="cuda", enabled=args.amp):
                    out = model(model_inputs)
                    loss = out["loss"] if isinstance(out, dict) and "loss" in out else out
                    loss = loss.mean()
            except torch.OutOfMemoryError as e:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise SystemExit(
                    f"CUDA OOM during forward pass at batch_size={args.batch_size}. "
                    f"Try lowering --batch-size (e.g. halve it) and increasing --grad-accum to keep effective batch. ({e})"
                ) from e
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    raise SystemExit(
                        f"CUDA OOM during forward pass at batch_size={args.batch_size}. "
                        f"Try lowering --batch-size (e.g. halve it) and increasing --grad-accum to keep effective batch. ({e})"
                    ) from e
                raise

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at global_step={global_step}: {loss.item()}")

            try:
                scaler.scale(loss).backward()
            except torch.OutOfMemoryError as e:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise SystemExit(
                    f"CUDA OOM during backward pass at batch_size={args.batch_size}. "
                    f"Try lowering --batch-size and increasing --grad-accum. ({e})"
                ) from e
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    raise SystemExit(
                        f"CUDA OOM during backward pass at batch_size={args.batch_size}. "
                        f"Try lowering --batch-size and increasing --grad-accum. ({e})"
                    ) from e
                raise

            if global_step % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    scaler.unscale_(opt)
                    last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                opt_step += 1

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
            should_log = args.log_every > 0 and (global_step % args.log_every == 0 or step + 1 == steps_per_epoch)
            if should_log:
                lr = opt.param_groups[0].get("lr", None)
                extra = ""
                if args.debug:
                    extra = f" encode={encode_time:.2f}s to_device={to_device_time:.2f}s"
                    if last_grad_norm is not None:
                        extra += f" grad_norm={last_grad_norm:.3f}"
                    if args.amp:
                        extra += f" scale={float(scaler.get_scale()):.1f}"
                    extra += f" {_cuda_mem()}"
                print(
                    f"[{_ts()}] [{bar}] {pct:5.1f}% "
                    f"[epoch {epoch+1}/{args.epochs} step {step+1}/{steps_per_epoch} "
                    f"(opt_step {opt_step_idx}/{total_opt_steps})] "
                    f"accum={accum_state} loss={loss.item():.4f} lr={lr} step_time={iter_time:.2f}s "
                    f"{eta_txt}{extra}"
                )
            if args.trace_steps:
                fwd_time = iter_time - data_time
                print(
                    f"[{_ts()}] [trace] step {step+1}: data_time={data_time:.3f}s, "
                    f"encode_time={encode_time:.3f}s, to_device_time={to_device_time:.3f}s, "
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
                    "data_path_norm": _norm_path(args.data),
                    "action_space": str(action_space),
                    "action_dim": int(getattr(cfg.model_cfg, "action_dim", 0) or 0),
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "step_in_epoch": step,
                    "global_step": global_step,
                    "opt_step": opt_step,
                    "args": vars(args),
                }
                safe_save_cache(cache_state, cache_path)

        cfg_to_save = cfg.model_copy(deep=True)
        try:
            cfg_to_save.tokenizer_cfg.training = True
        except Exception:
            pass
        try:
            cfg_to_save.tokenizer_cfg.use_action_mask = True
        except Exception:
            pass
        args.out.parent.mkdir(parents=True, exist_ok=True)
        finetune_epoch = int(epoch + 1)
        total_epoch = int(base_epoch + finetune_epoch)
        total_step = int(base_step + opt_step)
        payload = {
            "model": model.state_dict(),
            "step": total_step,
            "epoch": total_epoch,
            "ckpt_config": cfg_to_save.model_dump(),
            "finetune": {
                "epoch": int(finetune_epoch),
                "global_step": int(global_step),
                "opt_step": int(opt_step),
                "grad_accum": int(args.grad_accum),
                "steps_per_epoch": int(steps_per_epoch),
                "base_step": int(base_step),
                "base_epoch": int(base_epoch),
                "action_space": str(action_space),
            },
        }
        if action_space == "keyboard" and button_names_to_save:
            payload["button_names"] = list(button_names_to_save)
            payload["action_space"] = "keyboard"
        else:
            payload["action_space"] = "keyboard"
        _atomic_torch_save(payload, args.out)
        print(f"[+] Epoch {epoch+1} finished. Saved checkpoint to {args.out} (epoch={total_epoch} step={total_step})")


if __name__ == "__main__":
    main()
