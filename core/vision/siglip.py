import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from .base import BaseVisionEncoder

class VisionStateEncoder(BaseVisionEncoder):
    """
    Fast, continuous conversion of pixels into a STRUCTURED SEMANTIC ABSTRACTION.
    Uses real HuggingFace Vision backends (like SigLIP2) to extract embeddings
    and spatial features.
    """
    def __init__(self, target_resolution=(256, 256), vision_encoder_name="google/siglip-large-patch16-256", device="cuda"):
        self.target_resolution = target_resolution
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[Vision] Loading Production Vision Backbone: {vision_encoder_name}")

        self.processor = AutoImageProcessor.from_pretrained(vision_encoder_name, use_fast=True)
        self.vision_model = AutoModel.from_pretrained(vision_encoder_name)

        # Optimize model
        self.vision_model.eval()
        self.vision_model.to(self.device, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    @torch.inference_mode()
    def encode(self, raw_frame: np.ndarray) -> dict:
        """
        Takes a raw BGR/RGB frame and outputs compressed embeddings and semantic states.
        """
        # We explicitly ensure target_resolution is set to (256, 256) as required by NitroGen Policy.
        # Note: cv2.resize expects (width, height), shape is (height, width)
        if raw_frame.shape[0] != self.target_resolution[1] or raw_frame.shape[1] != self.target_resolution[0]:
            small_frame = cv2.resize(raw_frame, self.target_resolution, interpolation=cv2.INTER_AREA)
        else:
            small_frame = raw_frame

        # 1. Run through actual Vision Backbone to get spatial features
        pixel_values = self.processor(images=small_frame, return_tensors="pt")["pixel_values"].to(
            device=self.device, dtype=self.vision_model.dtype
        )
        vision_outputs = self.vision_model(pixel_values)

        # 2. Extract pooled representation
        pooled_output = vision_outputs.pooler_output.squeeze().cpu().numpy()
        last_hidden_state = vision_outputs.last_hidden_state.squeeze().cpu().numpy()

        # In a fully trained pipeline, this last_hidden_state is mapped to discrete JSON.
        # For the active runtime, we pass the real embeddings to the Policy Engine.

        # DYNAMIC SEMANTIC STATE EVALUATION
        # In a real system without a dedicated MLP, we can use simple statistical
        # variance on the pooled_output to detect significant screen changes.
        activity_level = np.var(pooled_output)

        if activity_level > 1.5:
            danger = "high"
            boss_act = "attacking"
        elif activity_level > 0.8:
            danger = "medium"
            boss_act = "windup"
        else:
            danger = "low"
            boss_act = "idle"

        semantic_state = {
            # Real tensor features ready for policy
            "raw_pixels_compressed": small_frame,
            "vision_embeddings": pooled_output,
            "hidden_states": last_hidden_state,
            # Dynamically extracted proxies for the LLM Brain
            "danger": danger,
            "boss_action": boss_act,
            "player_state": "active",
            "safe_window": (boss_act == "idle"),
            "geometry": {"activity_variance": float(activity_level)}
        }

        state = {
            "raw_pixels_compressed": small_frame,
            "semantic_state": semantic_state,
            "status": "combat_active"
        }
        return state
