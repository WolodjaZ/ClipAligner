import torch
from typing import Any, List

class LossNotImplementedError(NotImplementedError):
    """Raised when the loss function is not implemented."""
    pass

class BaseImageLoss(torch.nn.Module):
    """
    Base class for image loss functions in ClipAligner.
    """
    def forward(self, image_features: torch.Tensor, target_features: torch.Tensor, *args, **kwargs) -> torch.Tensor | List[torch.Tensor]:
        raise LossNotImplementedError(f"Forward method not implemented in {self.name}")
    
    def __repr__(self):
        key_values = [f"{key}={value}" for key, value in vars(self).items()]
        return f"{self.__class__.__name__}: {', '.join(key_values)}"



class BaseCaptionLoss(torch.nn.Module):
    """
    Base class for caption loss functions in ClipAligner.
    """

    def forward(self, text_features: torch.Tensor, target_features: torch.Tensor, *args, **kwargs) -> torch.Tensor | List[torch.Tensor]:
        raise LossNotImplementedError(f"Forward method not implemented in {self.name}")

    def __repr__(self):
        key_values = [f"{key}={value}" for key, value in vars(self).items()]
        return f"{self.__class__.__name__}: {', '.join(key_values)}"


class BaseImageCaptionLoss(torch.nn.Module):
    """
    Base class for combined image and caption loss functions in ClipAligner.
    """

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor, *args, **kwargs) -> torch.Tensor | List[torch.Tensor]:
        raise LossNotImplementedError(f"Forward method not implemented in {self.name}")

    def __repr__(self):
        key_values = [f"{key}={value}" for key, value in vars(self).items()]
        return f"{self.__class__.__name__}: {', '.join(key_values)}"

