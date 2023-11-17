from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from .base import BaseImageCaptionDataset, BaseImageDataset, BaseCaptionDataset, DatasetNotImplementedError
from .ccm import CC3MDataset, CC12MDataset
from .laion import Laion400Dataset


def get_dataset(cfg: dict | DictConfig) -> BaseImageDataset | BaseCaptionDataset | BaseImageCaptionDataset:
    """Get the dataset.
    
    Raises:
        DatasetNotImplementedError: If the loss function is not implemented.
    
    Args:
        cfg (DictConfig | dict): Configuration file containing "name" and "dataset_path" key and other parameters.
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
    
    # Get the dataset based on the name
    if name == "cc3m":
        return CC3MDataset(dataset_path, **cfg)
    elif name == "cc12m":
        return CC12MDataset(dataset_path, **cfg)
    elif name == "laion400":
        return Laion400Dataset(dataset_path, **cfg)
    else:
        raise DatasetNotImplementedError(f"Dataset {name} is not implemented.")
