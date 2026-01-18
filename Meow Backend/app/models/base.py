from abc import ABC, abstractmethod
from PIL import Image
from typing import Dict, Any, Tuple

class FERModel(ABC):
    """
    Abstract Base Class for all FER models.
    Every new model (ResNet, BLIP-2, LLaVA) MUST inherit from this.
    """

    @abstractmethod
    def load(self, device: str):
        """Load model weights into memory."""
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference.
        Must return a dictionary with:
        - label (str)
        - confidence (float)
        - explanation (str, optional)
        """
        pass