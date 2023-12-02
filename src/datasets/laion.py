import torch
from pathlib import Path

from .base import BaseImageCaptionDataset, DatasetNotImplementedError


class Laion400Dataset(BaseImageCaptionDataset):
    def __init__(self, dataset_path: Path, *args, **kwargs) -> None:
        super().__init__(dataset_path, *args, **kwargs)
        raise DatasetNotImplementedError("Dataset laion400 not yet implemented.")