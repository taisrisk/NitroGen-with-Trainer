from abc import ABC, abstractmethod
import numpy as np

class BaseVisionEncoder(ABC):
    """
    Abstract Base Class for Vision Encoders.
    Ensures that regardless of the backend (SigLIP, MobileSAM, DINOv3),
    the policy engine receives a standardized semantic state dictionary.
    """

    @abstractmethod
    def encode(self, raw_frame: np.ndarray) -> dict:
        """
        Input: Raw RGB/BGR pixel array from the screen capture.
        Output: Structured dictionary containing raw downscaled pixels,
                high-dimensional vision embeddings, and extracted semantic proxies.
        """
        pass
