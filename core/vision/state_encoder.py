from .base import BaseVisionEncoder

class VisionStateEncoderFactory:
    """
    Factory to construct the appropriate Vision Encoder based on config.
    Supports SigLIP2, DINOv3, and MobileSAM.
    """
    @staticmethod
    def create(backend="siglip", target_resolution=(256, 256), device="cuda") -> BaseVisionEncoder:
        if backend == "siglip":
            from .siglip import VisionStateEncoder
            return VisionStateEncoder(target_resolution=target_resolution, device=device)
        elif backend == "dinov3":
            raise NotImplementedError("DINOv3 backend not yet implemented.")
        elif backend == "mobilesam":
            raise NotImplementedError("MobileSAM backend not yet implemented.")
        else:
            raise ValueError(f"Unknown vision backend: {backend}")
