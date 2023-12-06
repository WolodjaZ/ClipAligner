from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from functools import partial
from loguru import logger
from torchvision import transforms
from typing import Callable, Dict, Optional, Tuple

from .base import BaseImageCaptionDataset, BaseImageDataset, BaseCaptionDataset, DatasetNotImplementedError
from .dumb import DumbImageDataset, DumbCaptionDataset, DumbImageCaptionDataset
from .ccm import CC3MDataset, CC12MDataset
from .laion import Laion400Dataset

try:
    import transformers # type: ignore
    tokenizer_type_slow = transformers.PreTrainedTokenizer
    tokenizer_type_fast = transformers.PreTrainedTokenizerFast
    tokenizer_type = tokenizer_type_slow | tokenizer_type_fast
except ImportError:
    from typing import Any
    tokenizer_type = Any
    tokenizer_type_slow = tokenizer_type_fast = None


def get_dataset(cfg: dict | DictConfig, transfor: Dict[str, Callable] | None = None) -> Tuple[BaseImageDataset, Optional[BaseImageDataset]] | Tuple[BaseCaptionDataset, Optional[BaseCaptionDataset]] | Tuple[BaseImageCaptionDataset, Optional[BaseImageCaptionDataset]]:
    """Get the dataset.
    
    Raises:
        DatasetNotImplementedError: If the loss function is not implemented.
    
    Args:
        cfg (DictConfig | dict): Configuration file containing "name" and "dataset_path" key and other parameters.
        transfor (Dict[str, Callable] | None): Transforms to apply to the dataset.
    Returns:
        (BaseImageDataset, BaseImageDataset | None) | (BaseCaptionDataset, BaseCaptionDataset | None) | (BaseImageCaptionDataset, BaseImageCaptionDataset | None): Dataset for train and validation.
    """
    # Transform the cfg to dictionary
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    # Extract the name, dataset path, text_max_length if provided
    name = cfg.pop("name", None)
    dataset_path = cfg.pop("dataset_path", None)
    max_length = cfg.pop("text_max_length", None)

    # Check transforms dict
    if transfor is not None:
        image_transform = transfor.get("image")
        caption_transform = transfor.get("caption")
        caption_transform = (
            set_tokenizer(caption_transform, max_length=max_length)
            if (isinstance(caption_transform, (Callable, tuple)))
            and caption_transform is not None
            else caption_transform
        )
    else:
        image_transform, caption_transform = None, None

    caption_transform_name = f"{type(caption_transform)} of {type(caption_transform.func)}" if isinstance(caption_transform, partial) else type(caption_transform)
    logger.debug(f"Transforms should be provided as a dict with keys 'image' or 'caption'. Provided transforms image {type(image_transform)} and caption {caption_transform_name}")

    # Validate the name and dataset path
    if name is None:
        raise DatasetNotImplementedError("Dataset name is not provided.")
    if dataset_path is None:
        raise DatasetNotImplementedError("Dataset path is not provided.")

    # Convert to Path
    dataset_path = Path(dataset_path)

    name = name.lower()
    # Get the dataset based on the name. The Image or Caption datasets
    if name == "dumb_image":
        image_transform = transforms.Compose(image_transform)
        return DumbImageDataset(dataset_path, transform=image_transform, **cfg), None
    elif name == "dumb_caption":
        return DumbCaptionDataset(dataset_path, transform=caption_transform, **cfg), None
    elif name == "dumb_image_caption":
        image_transform = transforms.Compose(image_transform)
        return DumbImageCaptionDataset(dataset_path, image_transform=image_transform, caption_transform=caption_transform, **cfg), None
    elif name == "cc3m":
        image_transform = transforms.Compose(image_transform.append(transforms.ToTensor())) if isinstance(image_transform, list) else image_transform
        return CC3MDataset(dataset_path, image_transform=image_transform, caption_transform=caption_transform, **cfg), None
    elif name == "cc12m":
        image_transform = transforms.Compose(image_transform.append(transforms.ToTensor())) if isinstance(image_transform, list) else image_transform
        return CC12MDataset(dataset_path, image_transform=image_transform, caption_transform=caption_transform, **cfg), None
    elif name == "laion400":
        image_transform = transforms.Compose(image_transform.append(transforms.ToTensor())) if isinstance(image_transform, list) else image_transform
        return Laion400Dataset(dataset_path, image_transform=image_transform, caption_transform=caption_transform, **cfg), None
    else:
        raise DatasetNotImplementedError(f"Dataset {name} is not implemented.")


def set_tokenizer(tokenizer: tokenizer_type, max_length: int | None = None) -> tokenizer_type:
    """Set the tokenizer.
    
    Args:
        tokenizer (tokenizer_type): Tokenizer.
        max_length (int | None): Maximum length of the text.
    Returns:
        tokenizer_type: Tokenizer.
    """
    return partial(tokenizer, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt") if max_length is not None else partial(tokenizer, padding=True, truncation=True, return_tensors="pt")