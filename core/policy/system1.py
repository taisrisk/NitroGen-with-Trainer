import time
import numpy as np
import torch

from nitrogen.inference_session import load_model
from nitrogen.mm_tokenizers import NitrogenTokenizer

class FastPolicyExecutor:
    """
    System 1: The fast motor execution layer.

    Loads actual PyTorch weights (NitroGen DiT) and directly maps the
    Vision Embs + LLM Intent to KBM Hardware actions.
    """
    def __init__(self, ckpt_path=None, device="cuda"):
        self.ckpt_path = ckpt_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

        if ckpt_path is not None:
            print(f"[Policy] Loading Production NitroGen Policy Model from {ckpt_path}...")
            try:
                # Load the full Nitrogen architecture properly
                self.model, self.tokenizer, self.img_proc, self.ckpt_config, _, _, _, _, _, _ = load_model(
                    ckpt_path,
                    device=self.device,
                    weights_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    old_layout=False
                )
                self.model.eval()
            except Exception as e:
                print(f"[Policy] Warning: Could not load model: {e}")

    @torch.inference_mode()
    def decide_hierarchical_action(self, semantic_state: dict, current_intent: str) -> dict:
        """
        Policy mapping: Real Vision Embeddings + Intent -> Real Hardware Action Tensors
        """
        if self.model is None or self.tokenizer is None:
            # Fallback to pure dummy logic if no weights were provided
            return self._fallback_logic(current_intent)

        # In a fully integrated flow, the NitroGen model receives the `current_intent`
        # mapped to its `game_id` condition space, and the `semantic_state` embeddings.
        # This executes the real Flow-Matching Denoising step.

        # Format dummy batch to fulfill Nitrogen tokenizer structure
        # (This avoids crashing the DiT while mapping to real outputs)
        batch_size = 1

        # Format the actual frame for the DiT.
        # Ensure it has the shape (Batch, Frames, C, H, W).
        # We take the compressed_pixels (H, W, C) from the Vision layer and transpose it.
        try:
            frame_hwc = semantic_state.get("raw_pixels_compressed")
            if frame_hwc is None:
                raise ValueError("No visual frame provided to policy.")

            # Defensive check: The DiT model implicitly expects exactly 256x256 dimensions.
            if frame_hwc.shape[0] != 256 or frame_hwc.shape[1] != 256:
                import cv2
                frame_hwc = cv2.resize(frame_hwc, (256, 256), interpolation=cv2.INTER_AREA)

            frame_chw = np.transpose(frame_hwc, (2, 0, 1)) # (H,W,C) -> (C,H,W)
            frame_seq = np.expand_dims(frame_chw, axis=0)  # (1, C, H, W)
            frame_batch = np.expand_dims(frame_seq, axis=0) # (1, 1, C, H, W)

            # Use '0' (Unconditional) as a safe fallback if the LLM string isn't explicitly
            # in the model's game_mapping dictionary, ensuring the DiT executes regardless.
            safe_intent = current_intent
            if self.tokenizer.game_mapping and current_intent not in self.tokenizer.game_mapping:
                safe_intent = None

            data_payload = {
                "images": frame_batch,
                "dropped_frames": torch.zeros((batch_size,), dtype=torch.bool),
                "game": safe_intent
            }

            tokenized_data = self.tokenizer.encode(data_payload)
            for k, v in tokenized_data.items():
                if isinstance(v, torch.Tensor):
                    tokenized_data[k] = v.to(self.device).unsqueeze(0)

            model_output = self.model.get_action(tokenized_data, old_layout=False)
            predicted_actions = self.tokenizer.decode(model_output)

            # Translate NitroGen raw tensors back to dictionary format
            j_left = predicted_actions["j_left"].squeeze().cpu().numpy()
            j_right = predicted_actions["j_right"].squeeze().cpu().numpy()
            buttons = predicted_actions["buttons"].squeeze().cpu().numpy()

            return {
                "type": "neural",
                "j_left": j_left,
                "j_right": j_right,
                "buttons": buttons
            }
        except Exception as e:
            # Absolute worst-case fallback to prevent game loop crash
            print(f"[Policy] Neural Execution Error: {e}")
            return self._fallback_logic(current_intent)

    def _fallback_logic(self, current_intent: str):
        if "ATTACK" in current_intent:
            return {"type": "heuristic", "action": "ATTACK_MELEE"}
        if "DODGE" in current_intent:
            return {"type": "heuristic", "action": "DODGE_BACK"}
        return {"type": "heuristic", "action": "WAIT"}

    def translate_to_hardware(self, hierarchical_action: dict) -> dict:
        """
        Translates the high-level or neural action into deterministic KBM hardware events.
        """
        hardware_action = {
            "mouse_dx": 0,
            "mouse_dy": 0,
            "keys_down": set(),
            "lmb": False,
            "rmb": False
        }

        act_type = hierarchical_action.get("type", "heuristic")

        if act_type == "neural":
            # Map continuous joystick outputs to discrete keys
            jl = hierarchical_action.get("j_left", [0, 0])
            if jl[1] < -0.2: hardware_action["keys_down"].add("w")
            elif jl[1] > 0.2: hardware_action["keys_down"].add("s")
            if jl[0] < -0.2: hardware_action["keys_down"].add("a")
            elif jl[0] > 0.2: hardware_action["keys_down"].add("d")

            # Map right stick to mouse deltas
            jr = hierarchical_action.get("j_right", [0, 0])
            hardware_action["mouse_dx"] = int(jr[0] * 50)
            hardware_action["mouse_dy"] = int(jr[1] * 50)

            # Map buttons
            btns = hierarchical_action.get("buttons", [])
            if len(btns) > 0 and btns[0] > 0.5: # Assuming index 0 is mapped to LMB
                hardware_action["lmb"] = True

        elif act_type == "heuristic":
            act = hierarchical_action.get("action", "WAIT")
            if act == "ATTACK_MELEE":
                hardware_action["lmb"] = True
            elif act == "DODGE_BACK":
                hardware_action["keys_down"].update(["s", "space"])

        return hardware_action
