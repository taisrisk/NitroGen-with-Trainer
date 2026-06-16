<img src="assets/github_banner.gif" width="100%" />

<div align="center">
  <p style="font-size: 1.2em;">
    <a href="https://nitrogen.minedojo.org/"><strong>Website</strong></a> | 
    <a href="https://huggingface.co/nvidia/NitroGen"><strong>Model</strong></a> |
    <a href="https://huggingface.co/datasets/nvidia/NitroGen"><strong>Dataset</strong></a> |
    <a href="https://nitrogen.minedojo.org/assets/documents/nitrogen.pdf"><strong>Paper</strong></a>
  </p>
</div>

# NitroGen - Modular Vision-Language-Action (VLA) Framework

NitroGen has evolved from a monolithic gaming model into a generalized, modular **Cognitive Architecture** optimized for real-time computer control, extremely low latency inference, and sample-efficient learning.

## 🧠 The 4-Pillar Cognitive Architecture

Instead of a brittle pixel-to-keyboard pipeline, NitroGen now utilizes a decoupled framework:

1. **Vision Layer (`core/vision`)**: A fast semantic state encoder (powered by models like SigLIP2) that compresses raw screens into structured representations (e.g. tracking "danger", "safe_windows", and "target_coords"). This eliminates the massive data requirements of raw pixel training.
2. **Brain Layer (`core/brain`)**: The **System 2** reasoning layer. An asynchronous micro-LLM (like Qwen2.5:3B via Ollama) that reads the semantic state and generates high-level `INTENT` macros (e.g. `STRATEGY: KITE_AND_HEAL`). It operates at 1-2 Hz without blocking motor execution.
3. **Policy Layer (`core/policy`)**: The **System 1** motor execution layer. Evaluates the NitroGen Diffusion Transformer against the Vision Embeddings and the Brain's Intent to output high-frequency, hierarchical Keyboard and Mouse actions at 30+ FPS.
4. **Episodic Memory (`core/memory`)**: Powered by **HelixDB**. Logs specific failure states (with explicit "critical factor" tags) and successful executions. The LLM queries this database in real-time to avoid repeating mistakes, allowing the agent to learn in 20-100 tries instead of thousands.

## ⚡ Latency Optimization & N-1 Queues

The agent loop (`scripts/run_agent.py`) is driven by the `AgentPipeline` which utilizes a **4-thread asynchronous queue architecture** (Capture → Vision → Policy → Execution).
Queues are hard-capped at `maxsize=1` (N-1 logic). If the neural networks lag, stale frames are immediately dropped to guarantee the policy always evaluates the absolute freshest screen state, resulting in ultra-low input latency.

## 🎛️ Dynamic Quantization & VRAM Management

Consumer hardware deployment is managed via `nitrogen/quantization.py`. The framework uses `pynvml` to dynamically detect available VRAM and assigns the optimal `BitsAndBytesConfig`:
- **< 8GB VRAM:** `NF4` + Double Quantization
- **< 16GB VRAM:** `INT8`
- **> 16GB VRAM:** `BFloat16` (or `FP16`)

## 📦 Model Export Utilities

Trained action policies can be exported to multiple formats for deployment in different high-speed runtimes:
- `python scripts/export/export_safetensors.py --ckpt ng.pt --out model.safetensors`
- `python scripts/export/export_onnx.py --ckpt ng.pt --out model.onnx`
- `python scripts/export/export_gguf.py --ckpt model_dir --out model.gguf`

---

## 🛠️ Installation

```bash
git clone https://github.com/MineDojo/NitroGen.git
cd NitroGen
pip install -e .
pip install ollama helix-db bitsandbytes pynvml
```

Make sure you have [Ollama](https://ollama.com/) installed and running on your local machine to support the System 2 Brain.

## 🚀 Running the Agent

Start the universal agent by invoking the run script:

```bash
python scripts/run_agent.py --process '<game_executable_name>.exe' --ckpt ng.pt
```

All agent configurations (target FPS, LLM context size, resolution) are centrally managed in `config/system.yaml`.

---

## Custom Training (train.py)

Fine-tune the policy layer on custom hierarchical actions.

### Data assumptions
- `obs`: `(N, 3, H, W)` float in `[0,1]` (channels-first frames).
- `actions`: `(N, T, 25)` float, mapping (buttons, lx, ly, rx, ry).
- `strategies`: A temporal array aligning LLM intent strings to specific frame chunks.

### Quick start
```bash
python scripts/train.py \
  --base-ckpt ng.pt \
  --data my_game_nitro.pt \
  --out my_game_finetune.pt \
  --epochs 3 \
  --batch-size 4 \
  --amp
```

### Key arguments
- `--base-ckpt`: NitroGen base checkpoint (`ckpt_config` + `model` keys). Required.
- `--data`: Path to your `*_nitro.pt` dataset. Required.
- `--out`: Where to write the fine-tuned checkpoint. Required.
- `--amp`: Enable mixed precision (recommended on modern GPUs).
- `--preencode`: Tokenize all samples once on CPU; lowers per-step overhead, uses disk cache.
- `--ultra-fast`: Enables TF32 where available and cuDNN benchmark.
