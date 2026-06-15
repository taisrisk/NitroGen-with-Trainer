import argparse
import torch
from nitrogen.inference_session import load_model
import numpy as np

def export_onnx(ckpt_path: str, out_path: str):
    """
    Exports the PyTorch NitroGen model to ONNX for highly optimized inference
    across TensorRT and DirectML.
    """
    print(f"Loading NitroGen PyTorch model from {ckpt_path}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, tokenizer, _, _, _, _, _, _, _, _ = load_model(
            ckpt_path,
            device=device,
            weights_dtype=torch.float32,
            old_layout=False
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # Create dummy inputs that match the NitroGen Tokenizer expectations
    batch_size = 1
    # Example input matches the required structure in `system1.py`
    dummy_data = {
        "images": np.zeros((batch_size, 1, 3, 256, 256), dtype=np.uint8),
        "dropped_frames": torch.zeros((batch_size,), dtype=torch.bool),
        "game": "INTENT: WAIT"
    }

    tokenized_data = tokenizer.encode(dummy_data)
    for k, v in tokenized_data.items():
        if isinstance(v, torch.Tensor):
            tokenized_data[k] = v.to(device).unsqueeze(0)

    # Note: Torch ONNX exporter requires unpacking dictionary arguments if the forward pass doesn't naturally accept dicts.
    # Since NitroGen's forward/get_action explicitly accepts `data` as a dict, we wrap it into a single tuple.
    dummy_input = (tokenized_data,)

    print(f"Exporting to ONNX -> {out_path}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            out_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_dict'],
            output_names=['action_output'],
            dynamic_axes={'input_dict': {0: 'batch_size'}, 'action_output': {0: 'batch_size'}}
        )
        print("ONNX Export complete.")
    except Exception as e:
        print(f"ONNX Export Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Input checkpoint path (.pt)")
    parser.add_argument("--out", type=str, required=True, help="Output ONNX path")
    args = parser.parse_args()
    export_onnx(args.ckpt, args.out)
