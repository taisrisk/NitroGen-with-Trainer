import torch
import gc
import pynvml
from transformers import BitsAndBytesConfig

def get_gpu_vram_gb(device_idx=0):
    """Detect available VRAM for automatic quantization selection."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.free / (1024 ** 3)
    except Exception:
        # Fallback if pynvml is unavailable
        if torch.cuda.is_available():
            free_mem, _ = torch.cuda.mem_get_info(device_idx)
            return free_mem / (1024 ** 3)
        return 0

def auto_select_quantization(requested_mode="auto"):
    """
    Dynamically select the best precision format to maximize performance
    while preventing OOM crashes on consumer GPUs.
    """
    if not torch.cuda.is_available() or requested_mode == "none":
        return None

    vram_gb = get_gpu_vram_gb()

    if requested_mode == "auto":
        if vram_gb < 8.0:
            mode = "nf4"
        elif vram_gb < 16.0:
            mode = "int8"
        else:
            mode = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    else:
        mode = requested_mode

    print(f"[Quantization] Detected {vram_gb:.1f} GB Free VRAM. Selected mode: {mode.upper()}")
    return get_bnb_config(mode)

def get_bnb_config(mode: str) -> BitsAndBytesConfig:
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if mode == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype
        )
    elif mode == "int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_has_fp16_weight=False
        )
    return None

def apply_mixed_precision(model):
    """
    If running without strict BitsAndBytes 4/8-bit quant, cast
    layers to optimal half-precision formats for inference speed.
    """
    if not torch.cuda.is_available():
        return model

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.to(dtype)
    return model
