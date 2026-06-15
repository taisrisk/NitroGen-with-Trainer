import argparse
import torch
from safetensors.torch import save_file

def export_safetensors(ckpt_path: str, out_path: str):
    """
    Exports a raw PyTorch Nitogen checkpoint into SafeTensors format
    for fast, secure loading across environments.
    """
    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model" not in checkpoint:
        raise ValueError("Invalid checkpoint format. Expected 'model' key.")

    state_dict = checkpoint["model"]

    print(f"Exporting to {out_path}...")
    save_file(state_dict, out_path)
    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Input checkpoint path (.pt)")
    parser.add_argument("--out", type=str, required=True, help="Output safetensors path (.safetensors)")
    args = parser.parse_args()
    export_safetensors(args.ckpt, args.out)
