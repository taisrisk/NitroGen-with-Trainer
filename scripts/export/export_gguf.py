import argparse
import subprocess
import os
import sys

def export_gguf(ckpt_path: str, out_path: str):
    """
    Automates the conversion of the underlying LLM backbones to GGUF format
    by leveraging the local llama.cpp converter script.
    """
    # Assuming llama.cpp has been cloned to the local workspace
    llama_cpp_path = os.path.join(os.getcwd(), "llama.cpp")
    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")

    if not os.path.exists(convert_script):
        print(f"[Error] llama.cpp conversion script not found at {convert_script}.")
        print("Please run `git clone https://github.com/ggml-org/llama.cpp.git` in the root directory first.")
        sys.exit(1)

    print(f"Executing native llama.cpp GGUF conversion for {ckpt_path}...")

    # We target the HuggingFace compatible output directory that NitroGen produces
    cmd = [
        sys.executable,
        convert_script,
        ckpt_path,
        "--outfile",
        out_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[Export Error] Conversion failed.")
        print(result.stderr)
    else:
        print(f"Successfully exported to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to HF model directory")
    parser.add_argument("--out", type=str, required=True, help="Output GGUF file path")
    args = parser.parse_args()
    export_gguf(args.ckpt, args.out)
