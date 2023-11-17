import torch
from pathlib import Path
from loguru import logger

from torch.utils.data import Dataset

class DatasetNotImplementedError(NotImplementedError):
    """Dataset not implemented error."""
    pass


class BaseImageDataset(Dataset):
    def __init__(self, dataset_path: Path, *args, **kwargs) -> None:
        super().__init__()
        self._dataset_path = dataset_path
        self.validate_dataset()
        self._image_paths = self._get_image_paths()
        logger.debug(f"Dataset Image {self.__class__.__name__} path: {self._dataset_path} | Number of images: {len(self)}")
    
    def __len__(self) -> int:
        return len(self._image_paths)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        raise NotImplementedError(f"Dataset {self.__class__.__name__} not implemented.")
    
    def _get_image_paths(self) -> list[Path]:
        raise NotImplementedError(f"Dataset {self.__class__.__name__} not implemented.")
    
    @staticmethod
    def validate_dataset(dataset_path: Path):
        if not dataset_path.exists():
            raise DatasetNotImplementedError(f"Dataset path {dataset_path} does not exist.")


class BaseCaptionDataset(Dataset):
    def __init__(self, dataset_path: Path, *args, **kwargs) -> None:
        super().__init__()
        self._dataset_path = dataset_path
        self.validate_dataset()
        self._captions = self._get_captions()
        logger.debug(f"Dataset Caption path: {self._dataset_path} | Number of captions: {len(self)}")
    
    def __len__(self) -> int:
        return len(self._captions)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        raise NotImplementedError("Dataset not implemented.")

    def _get_image_paths(self) -> list[Path]:
        raise NotImplementedError(f"Dataset {self.__class__.__name__} not implemented.")

    @staticmethod
    def validate_dataset(dataset_path: Path):
        if not dataset_path.exists():
            raise DatasetNotImplementedError(f"Dataset path {dataset_path} does not exist.")


class BaseImageCaptionDataset(Dataset):
    def __init__(self, dataset_path: Path, *args, **kwargs) -> None:
        super().__init__()
        self._dataset_path = dataset_path
        self.validate()
        self._image_paths = self._get_image_paths()
        logger.debug(f"Dataset ImageCaption path: {self._dataset_path} | Number of samples: {len(self)}")
    
    def __len__(self) -> int:
        return len(self._image_paths)
    
    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError("Dataset not implemented.")

    def _get_image_paths(self) -> list[Path]:
        raise NotImplementedError(f"Dataset {self.__class__.__name__} not implemented.")
    
    def _get_caption_for_image(self, image_path: Path) -> str:
        raise NotImplementedError("Dataset not implemented.")

    @staticmethod
    def validate_dataset(dataset_path: Path):
        if not dataset_path.exists():
            raise DatasetNotImplementedError(f"Dataset path {dataset_path} does not exist.")