from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from loguru import logger
from .base import BaseImageModel, BaseCaptionModel, BaseImageCaptionModel, ModelNotImplementedError
from .models import DinoVisionModel, RobertaCaptionModel
from .aligner import ClipAligner


SEPARATOR = "|"


def create_model(cfg: dict | DictConfig) -> BaseImageModel | BaseCaptionModel | BaseImageCaptionModel:
    """Create the model. This function is used to create the model based on the configuration file.
    
    Raises:
        ModelNotImplementedError: If the model is not implemented.
    
    Args:
        cfg (DictConfig | dict): Configuration file containing "name" and other parameters.
    Returns:
        BaseImageModel | BaseCaptionModel | BaseImageCaptionModel: Model.
    """
    # Transform the cfg to dictionary
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    # Extract the models
    image_cfg, caption_cfg, alignment_cfg = extract_models_config(cfg)

    # Get the name and checkpoints
    name = ""
    checkpoints = {}

    if image_cfg is not None:
        buff_name = image_cfg.pop("name")
        name += buff_name
        checkpoints[buff_name] = image_cfg.pop("checkpoint", None)
    if caption_cfg is not None:
        buff_name = caption_cfg.pop("name")
        name += SEPARATOR + buff_name
        checkpoints[buff_name] = caption_cfg.pop("checkpoint", None)
    if alignment_cfg is not None:
        buff_name = alignment_cfg.pop("name")
        name += SEPARATOR + buff_name
        # checkpoints[buff_name] = alignment_cfg.pop("checkpoint", None)

    # Get the dataset based on the name
    if name == "dinov2|roberta|clip_aligner":
        vision_model_name, caption_model_name, aligner_model_name = name.split(SEPARATOR)
        logger.info(f"Initializing Vision model: {vision_model_name} and Caption model: {caption_model_name}")
        vision_model = DinoVisionModel(checkpoint=checkpoints[vision_model_name], **image_cfg)
        caption_model = RobertaCaptionModel(checkpoint=checkpoints[caption_model_name], **caption_cfg)
        return ClipAligner(
            vision_model=vision_model,
            caption_model=caption_model,
            **alignment_cfg
        )
    else:
        raise ModelNotImplementedError(f"Model {name} is not implemented.")


def extract_models_config(cfg: dict) -> (dict | None, dict | None, dict | None):
    """Extract the models configuration.
    Models for Image, Caption, and Alignment will be extracted.

    Args:
        cfg (dict): Configuration file.
    Returns:
        dict | None: Image model configuration.
        dict | None: Caption model configuration.
        dict | None: Alignment model configuration.
    """
    image_cfg = cfg.pop("image", None)
    caption_cfg = cfg.pop("caption", None)
    alignment_cfg = cfg.pop("alignment", None)
    return image_cfg, caption_cfg, alignment_cfg