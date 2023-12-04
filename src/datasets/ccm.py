import io
import torch
import tarfile
from PIL import Image

from pathlib import Path
from loguru import logger
from typing import Callable

from .base import BaseImageCaptionDataset, DatasetNotImplementedError

class CC3MDataset(BaseImageCaptionDataset):
    def __init__(self, dataset_path: Path, image_transform: Callable | None = None, caption_transform: Callable | None = None, *args, **kwargs) -> None:
        super().__init__(dataset_path, image_transform, caption_transform, *args, **kwargs)
    
    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor | str):
        image_path = self._image_paths[index]
        with tarfile.open(image_path, "r") as tar:
            image_data = tar.extractfile("image.jpg").read()
            image = Image.open(io.BytesIO(image_data)) if self._image_transform is None else self._image_transform(Image.open(io.BytesIO(image_data)))
            caption = self._get_caption_for_image(image_path) if self._caption_transform is None else self._caption_transform(self._get_caption_for_image(image_path))
        
        return image, caption
    
    def _get_image_paths(self) -> list[Path]:
        return list(self._dataset_path.glob("*.tar"))
    
    def _get_caption_for_image(self, image_path: Path) -> torch.Tensor | str | dict:
        caption_path = image_path.with_suffix(".txt")
        with open(caption_path, "r") as f:
            caption = f.read()
        return caption
    
    @staticmethod
    def validate_dataset(dataset_path: Path):
        super().validate_dataset(dataset_path)
        if not dataset_path.is_dir():
            raise DatasetNotImplementedError(f"Dataset path {dataset_path} is not a directory needed for cc3m dataset.")

class CC12MDataset(BaseImageCaptionDataset):
    def __init__(self, dataset_path: Path, image_transform: Callable | None = None, caption_transform: Callable | None = None, *args, **kwargs) -> None:
        super().__init__(dataset_path, image_transform, caption_transform, *args, **kwargs)
        raise DatasetNotImplementedError("Dataset cc12m not yet implemented.")