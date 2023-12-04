from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Callable, Tuple

from .base import BaseImageCaptionDataset, BaseImageDataset, BaseCaptionDataset, DatasetNotImplementedError
from .dumb import DumbImageDataset, DumbCaptionDataset, DumbImageCaptionDataset
from .ccm import CC3MDataset, CC12MDataset
from .laion import Laion400Dataset


def get_dataset(cfg: dict | DictConfig, transforms: Callable | None | Tuple[Callable, Callable] = None) -> BaseImageDataset | BaseCaptionDataset | BaseImageCaptionDataset:
    """Get the dataset.
    
    Raises:
        DatasetNotImplementedError: If the loss function is not implemented.
    
    Args:
        cfg (DictConfig | dict): Configuration file containing "name" and "dataset_path" key and other parameters.
        transforms (callable | None | (callable, callable)): Transformations to be applied to the dataset.
    Returns:
         BaseImageDataset | BaseCaptionDataset | BaseImageCaptionDataset: Dataset.
    """
    # Transform the cfg to dictionary
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    
    # Extract the name and dataset path
    name = cfg.pop("name", None)
    dataset_path = cfg.pop("dataset_path", None)
    
    # Validate the name and dataset path
    if name is None:
        raise DatasetNotImplementedError("Dataset name is not provided.")
    if dataset_path is None:
        raise DatasetNotImplementedError("Dataset path is not provided.")
    
    # Convert to Path
    dataset_path = Path(dataset_path)
    
    # Get the dataset based on the name. The Image or Caption datasets
    if name == "dumb_image":
        return DumbImageDataset(dataset_path, transform=transforms, **cfg)
    elif name == "dumb_caption":
        return DumbCaptionDataset(dataset_path, transform=transforms, **cfg)
    
    # Get the transformations Image and Caption datasets
    if isinstance(transforms, Callable):
        raise ValueError("Image and caption transformations must be provided as a tuple or None.")
    elif isinstance(transforms, tuple):
        image_transform, caption_transform = transforms
    elif transforms is None:
        image_transform, caption_transform = None, None
    
    # Get the dataset based on the name. The Image and Caption datasets
    if name == "dumb_image_caption":
        return DumbImageCaptionDataset(dataset_path, image_transform=image_transform, caption_transform=caption_transform, **cfg)
    elif name == "cc3m":
        return CC3MDataset(dataset_path, image_transform=image_transform, caption_transform=caption_transform, **cfg)
    elif name == "cc12m":
        return CC12MDataset(dataset_path, image_transform=image_transform, caption_transform=caption_transform, **cfg)
    elif name == "laion400":
        return Laion400Dataset(dataset_path, image_transform=image_transform, caption_transform=caption_transform, **cfg)
    else:
        raise DatasetNotImplementedError(f"Dataset {name} is not implemented.")
