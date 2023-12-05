import torch
from pathlib import Path
from loguru import logger
from typing import Dict, Any

from torch import nn


class ModelNotImplementedError(NotImplementedError):
    """Model not implemented error."""


class BaseImageModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        logger.debug(f"Model image {self.__class__.__name__} initialized.")
    
    @property
    def output_dim(self) -> int:
        raise NotImplementedError(f"Model {self.__class__.__name__} not implemented.")
    
    def get_transformations(self) -> Dict[str, Any]:
        raise NotImplementedError(f"Model {self.__class__.__name__} not implemented.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"Model {self.__class__.__name__} not implemented.")


class BaseCaptionModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        logger.debug(f"Model caption {self.__class__.__name__} initialized.")
    
    @property
    def output_dim(self) -> int:
        raise NotImplementedError(f"Model {self.__class__.__name__} not implemented.")

    def get_transformations(self) -> Dict[str, Any]:
        raise NotImplementedError(f"Model {self.__class__.__name__} not implemented.")
    
    def forward(self, x: torch.Tensor | str | dict) -> torch.Tensor:
        raise NotImplementedError(f"Model {self.__class__.__name__} not implemented.")


class BaseImageCaptionModel(nn.Module):
    def __init__(self, vision_model: BaseImageModel, caption_model: BaseCaptionModel, *args, **kwargs) -> None:
        super().__init__()
        self._vision_model = vision_model
        self._caption_model = caption_model
        logger.debug(f"Model image-caption {self.__class__.__name__} initialized.")

    @property
    def output_dim(self) -> int:
        raise NotImplementedError(f"Model {self.__class__.__name__} not implemented.")

    def get_transformations(self) -> Dict[str, Any]:
        raise NotImplementedError(f"Model {self.__class__.__name__} not implemented.")
    
    def forward(self, image: torch.Tensor, caption: torch.Tensor | str | dict) -> torch.Tensor:
        raise NotImplementedError(f"Model {self.__class__.__name__} not implemented.")