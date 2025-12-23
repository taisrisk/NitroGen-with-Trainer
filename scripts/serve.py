import zmq
import argparse
import pickle
import gc
import time
import traceback
import sys

import torch
import numpy as np

from nitrogen.inference_session import InferenceSession


def _install_gc_logger():
    start_times_by_gen = {}

    def _cb(phase, info):
        gen = info.get("generation", "?")
        if phase == "start":
            start_times_by_gen[gen] = time.perf_counter()
            return

        t0 = start_times_by_gen.pop(gen, None)
        if t0 is None:
            return
        dt_s = time.perf_counter() - t0
        print(
            f"[gc] gen={gen} collected={info.get('collected')} "
            f"uncollectable={info.get('uncollectable')} dt={dt_s:.3f}s"
        )

    gc.callbacks.append(_cb)


def _select_amp_dtype(device: str, amp_dtype: str) -> torch.dtype | None:
    amp_dtype = amp_dtype.lower().strip()
    if not device.startswith("cuda"):
        return None

    if amp_dtype in ("none", "off", "fp32", "float32"):
        return None
    if amp_dtype in ("fp16", "float16", "half"):
        return torch.float16
    if amp_dtype in ("bf16", "bfloat16"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("[warn] bf16 not supported on this GPU; falling back to fp16")
        return torch.float16
    if amp_dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    raise ValueError(f"Unknown --amp-dtype: {amp_dtype!r}")


def _select_weights_dtype(device: str, weights_dtype: str) -> torch.dtype | None:
    weights_dtype = weights_dtype.lower().strip()
    if weights_dtype in ("fp32", "float32"):
        return None
    if not device.startswith("cuda") and weights_dtype in ("auto", "fp16", "float16", "half", "bf16", "bfloat16"):
        return None
    if weights_dtype in ("fp16", "float16", "half"):
        return torch.float16
    if weights_dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if weights_dtype == "auto":
        return torch.float16 if device.startswith("cuda") else None

    raise ValueError(f"Unknown --weights-dtype: {weights_dtype!r}")


def _cuda_device_index(device: str) -> int | None:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return None
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model inference server")
    parser.add_argument("ckpt", type=str, help="Path to checkpoint file")
    parser.add_argument("--port", type=int, default=5555, help="Port to serve on")
    parser.add_argument("--old-layout", action="store_true", help="Use old layout")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--ctx", type=int, default=1, help="Context length")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for model/inference (e.g. cuda, cuda:0, cuda:1)")
    parser.add_argument("--amp-dtype", type=str, default="auto", help="Autocast dtype: auto|fp16|bf16|none")
    parser.add_argument("--weights-dtype", type=str, default="auto", help="Model weights dtype: auto|fp16|bf16|fp32")
    parser.add_argument("--steps", type=int, default=None, help="Override model.num_inference_timesteps (flow-matching sampling steps)")
    parser.add_argument("--warmup", type=int, default=0, help="Run N warmup predicts at startup")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="Enable TF32 (faster on Ampere+)")
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True, help="Enable cuDNN benchmark autotuning")
    parser.add_argument("--gc-freeze", action=argparse.BooleanOptionalAction, default=True, help="Run gc.freeze() after model load")
    parser.add_argument("--gc-disable-auto", action=argparse.BooleanOptionalAction, default=True, help="Disable automatic GC (manual collection only)")
    parser.add_argument("--gc-collect-every", type=int, default=10, help="Run gc.collect() every N predictions (0 disables)")
    parser.add_argument("--cuda-empty-cache-every", type=int, default=0, help="Run torch.cuda.empty_cache() every N predictions (0 disables)")
    parser.add_argument("--gc-log", action="store_true", help="Log GC pause durations")
    args = parser.parse_args()

    print(f"[env] python={sys.executable}")
    print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "[error] --device=cuda requested but CUDA is unavailable in this Python environment. "
            "You likely ran the wrong interpreter (CPU-only torch). "
            "Use `./.venv/Scripts/python.exe scripts/serve.py ...` or pass `--device cpu`."
        )

    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    amp_dtype = _select_amp_dtype(args.device, args.amp_dtype)
    weights_dtype = _select_weights_dtype(args.device, args.weights_dtype)

    session = InferenceSession.from_ckpt(
        args.ckpt,
        old_layout=args.old_layout,
        cfg_scale=args.cfg,
        context_length=args.ctx,
        device=args.device,
        amp_dtype=amp_dtype,
        weights_dtype=weights_dtype,
    )
    if args.steps is not None and hasattr(session.model, "num_inference_timesteps"):
        session.model.num_inference_timesteps = int(args.steps)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        idx = _cuda_device_index(args.device)
        if idx is not None and idx < torch.cuda.device_count():
            cap = torch.cuda.get_device_capability(idx)
            name = torch.cuda.get_device_name(idx)
        else:
            cap = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
        print(f"[config] cuda_device={name} capability={cap} bf16_supported={torch.cuda.is_bf16_supported()}")
    print(f"[config] device={args.device} amp_dtype={amp_dtype} weights_dtype={weights_dtype} tf32={args.tf32} cudnn_benchmark={args.cudnn_benchmark}")
    if hasattr(session.model, "num_inference_timesteps"):
        print(f"[config] num_inference_timesteps={getattr(session.model, 'num_inference_timesteps', None)}")

    if args.gc_log:
        _install_gc_logger()

    gc.collect()
    if args.gc_freeze and hasattr(gc, "freeze"):
        gc.freeze()

    if args.gc_disable_auto:
        gc.disable()

    if args.warmup > 0:
        print(f"[warmup] running {args.warmup} predict(s)...")
        session.reset()
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        for i in range(args.warmup):
            if session.device.startswith("cuda"):
                torch.cuda.synchronize()
            session.predict(dummy)
            if session.device.startswith("cuda"):
                torch.cuda.synchronize()
            print(f"[warmup] {i + 1}/{args.warmup} done")
        session.reset()
        dt_s = time.perf_counter() - t0
        print(f"[warmup] total={dt_s:.3f}s avg={dt_s / args.warmup:.3f}s")

    # Setup ZeroMQ
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")

    # Create poller
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    print(f"\n{'='*60}")
    print(f"Server running on port {args.port}")
    print(f"Waiting for requests...")
    print(f"{'='*60}\n")

    predict_count = 0

    try:
        while True:
            # Poll with 100ms timeout to allow interrupt handling
            events = dict(poller.poll(timeout=100))
            if socket in events and events[socket] == zmq.POLLIN:
                # Receive request only when data is available
                request = socket.recv()
                request = pickle.loads(request)
                if request["type"] == "reset":
                    try:
                        session.reset()
                        response = {"status": "ok"}
                        print("Session reset")
                    except Exception as e:
                        traceback.print_exc()
                        response = {"status": "error", "message": f"reset failed: {e!r}"}
                elif request["type"] == "info":
                    try:
                        info = session.info()
                        response = {"status": "ok", "info": info}
                        print("Sent session info")
                    except Exception as e:
                        traceback.print_exc()
                        response = {"status": "error", "message": f"info failed: {e!r}"}
                elif request["type"] == "predict":
                    try:
                        raw_image = request["image"]
                        result = session.predict(raw_image)
                        response = {
                            "status": "ok",
                            "pred": result
                        }
                        predict_count += 1
                        run_gc = args.gc_collect_every > 0 and (predict_count % args.gc_collect_every == 0)
                        run_cuda_cache = args.cuda_empty_cache_every > 0 and (predict_count % args.cuda_empty_cache_every == 0)
                        if run_gc or run_cuda_cache:
                            t0 = time.perf_counter()
                            if run_gc:
                                gc.collect()
                            if run_cuda_cache and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            dt_s = time.perf_counter() - t0
                            if args.gc_log or dt_s >= 1.0:
                                print(f"[maintenance] n={predict_count} dt={dt_s:.3f}s (gc={run_gc}, cuda_empty_cache={run_cuda_cache})")
                    except Exception as e:
                        traceback.print_exc()
                        response = {"status": "error", "message": f"predict failed: {e!r}"}
                else:
                    response = {"status": "error", "message": f"Unknown request type: {request['type']}"}
                # Send response
                socket.send(pickle.dumps(response))
    except KeyboardInterrupt:
        print("\nShutting down server...")
        exit(0)
    finally:
        socket.close()
        context.term()
