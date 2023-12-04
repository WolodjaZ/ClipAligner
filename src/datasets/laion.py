import torch
from pathlib import Path
from typing import Callable

from .base import BaseImageCaptionDataset, DatasetNotImplementedError


class Laion400Dataset(BaseImageCaptionDataset):
    def __init__(self, dataset_path: Path, image_transform: Callable | None = None, caption_transform: Callable | None = None, *args, **kwargs) -> None:
        super().__init__(dataset_path, image_transform, caption_transform, *args, **kwargs)
        raise DatasetNotImplementedError("Dataset laion400 not yet implemented.")