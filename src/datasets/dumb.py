import torch
import string

from pathlib import Path
from loguru import logger
from typing import Callable

from .base import BaseImageDataset, BaseCaptionDataset, BaseImageCaptionDataset


SAMPLE_TEXT = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla a tortor neque. 
Suspendisse et pretium odio. Ut tempus enim nec ipsum euismod sagittis. Vivamus id arcu quis odio mollis maximus et sed erat. 
Nullam feugiat, eros sit amet porta luctus, eros sem mollis velit, vel sollicitudin sapien ipsum eget ipsum. Suspendisse vel dui sem. 
Interdum et malesuada fames ac ante ipsum primis in faucibus.
"""


class DumbImageDataset(BaseImageDataset):
    def __init__(self, dataset_path: Path, train_size: int, img_size: (int, int) = (64, 64), transform: Callable | None = None, *args, **kwargs) -> None:
        self._train_size = train_size
        super().__init__(dataset_path, transform, *args, **kwargs)
        self._img_size = img_size
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.rand(3, *self._img_size) if self._transform is None else self._transform(torch.rand(3, *self._img_size))
    
    def _get_image_paths(self) -> list[Path]:
        return [Path("dummy") for _ in range(self._train_size)]
    
    @staticmethod
    def validate_dataset(dataset_path: Path):
        pass


class DumbCaptionDataset(BaseCaptionDataset):
    def __init__(self, dataset_path: Path, train_size: int, text_max_length: int = 50, transform: Callable | None = None, *args, **kwargs) -> None:
        self._train_size = train_size
        super().__init__(dataset_path, transform, *args, **kwargs)
        self._text_max_length = text_max_length if text_max_length > 0 and text_max_length < len(SAMPLE_TEXT) else len(SAMPLE_TEXT)
    
    def __getitem__(self, index: int) -> torch.Tensor | str | dict:
        text_length = torch.randint(1, self._text_max_length, (1,)).item()
        return SAMPLE_TEXT[:text_length] if self._transform is None else self._transform(SAMPLE_TEXT[:text_length])
    
    def _get_caption_paths(self) -> list[Path]:
        return [Path("dummy") for _ in range(self._train_size)]
    
    @staticmethod
    def validate_dataset(dataset_path: Path):
        pass


class DumbImageCaptionDataset(BaseImageCaptionDataset):
    def __init__(self, dataset_path: Path, train_size: int, text_max_length: int = 50, img_size: (int, int) = (64, 64), image_transform: Callable | None = None, caption_transform: Callable | None = None, *args, **kwargs) -> None:
        self._train_size = train_size
        super().__init__(dataset_path, image_transform, caption_transform, *args, **kwargs)
        self._text_max_length = text_max_length if text_max_length > 0 and text_max_length < len(SAMPLE_TEXT) else len(SAMPLE_TEXT)
        self._image_size = tuple(img_size)

    def _get_image_paths(self) -> list[Path]:
        return [Path("dummy") for _ in range(self._train_size)]
    
    def _get_caption_for_image(self, image_path: Path) -> str:
        return ""
    
    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor | str | dict):
        img_tensor = torch.rand(3, *self._image_size) if self._image_transform is None else self._image_transform(torch.rand(3, *self._image_size))
        
        text_length = torch.randint(1, self._text_max_length, (1,)).item()
        caption = SAMPLE_TEXT[:text_length] if self._caption_transform is None else self._caption_transform(SAMPLE_TEXT[:text_length])
        return img_tensor, caption
    
    @staticmethod
    def validate_dataset(dataset_path: Path):
        pass