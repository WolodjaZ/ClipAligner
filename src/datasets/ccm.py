import torch
from PIL import Image

from pathlib import Path
from typing import Callable

from .base import BaseImageCaptionDataset, DatasetNotImplementedError

class CC3MDataset(BaseImageCaptionDataset):
    def __init__(self, dataset_path: Path, train_size: float, image_transform: Callable | None = None, caption_transform: Callable | None = None, *args, **kwargs) -> None:
        super().__init__(dataset_path, image_transform, caption_transform, *args, **kwargs)
        if train_size <= 0 or train_size > 1:
            raise ValueError(f"Train size must be between 0 and 1, got {train_size}.")
        self._train_size = train_size
        image_paths = self._get_image_paths()
        self._image_paths = image_paths[:int(len(image_paths) * train_size)]
        
        
    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor | str):
        image_path = self._image_paths[index]
        image = Image.open(image_path) if self._image_transform is None else self._image_transform(Image.open(image_path))
        caption = self._get_caption_for_image(image_path) if self._caption_transform is None else self._caption_transform(self._get_caption_for_image(image_path))
        return image, caption
    
    def _get_image_paths(self) -> list[Path]:
        return sorted(list(self._dataset_path.glob("*.jpg")), key=lambda x: int(x.stem))
    
    def _get_caption_for_image(self, image_path: Path) -> str:
        caption_path = image_path.with_suffix(".txt")
        with open(caption_path, "r") as f:
            caption = f.read()
        return caption
    
    @staticmethod
    def validate_dataset(dataset_path: Path):
        BaseImageCaptionDataset.validate_dataset(dataset_path)
        if not dataset_path.is_dir():
            raise DatasetNotImplementedError(f"Dataset path {dataset_path} is not a directory needed for cc3m dataset.")

class CC12MDataset(BaseImageCaptionDataset):
    def __init__(self, dataset_path: Path, image_transform: Callable | None = None, caption_transform: Callable | None = None, *args, **kwargs) -> None:
        super().__init__(dataset_path, image_transform, caption_transform, *args, **kwargs)
        raise DatasetNotImplementedError("Dataset cc12m not yet implemented.")