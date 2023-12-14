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
        buff_name = image_cfg.pop("name", None)
        if buff_name is None:
            raise ModelNotImplementedError("Image model name is not provided. Please provide the name in the configuration file or don't include it in configs.")
        buff_name = buff_name.lower()
        name += buff_name
        checkpoints[buff_name] = image_cfg.pop("checkpoint", None)
    if caption_cfg is not None:
        buff_name = caption_cfg.pop("name")
        if buff_name is None:
            raise ModelNotImplementedError("Caption model name is not provided. Please provide the name in the configuration file or don't include it in configs.")
        buff_name = buff_name.lower()
        name += SEPARATOR + buff_name
        checkpoints[buff_name] = caption_cfg.pop("checkpoint", None)
    if alignment_cfg is not None:
        buff_name = alignment_cfg.pop("name")
        if buff_name is None:
            raise ModelNotImplementedError("Alignment model name is not provided. Please provide the name in the configuration file or don't include it in configs.")
        buff_name = buff_name.lower()
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

def check_model_params(cfg_used: dict | DictConfig, cfg_loaded: dict | DictConfig) -> bool:
    """Check if the model parameters are the same.

    Args:
        cfg_used (dict | DictConfig): Configuration file used.
        cfg_loaded (dict | DictConfig): Configuration file loaded.
    Returns:
        bool: True if the parameters are the same, False otherwise.
    """
    # Transform the cfg to dictionary
    if isinstance(cfg_used, DictConfig):
        cfg_used = OmegaConf.to_container(cfg_used, resolve=True)
    if isinstance(cfg_loaded, DictConfig):
        cfg_loaded = OmegaConf.to_container(cfg_loaded, resolve=True)

    # Extract the models
    image_cfg_used, caption_cfg_used, alignment_cfg_used = extract_models_config(cfg_used)
    image_cfg_loaded, caption_cfg_loaded, alignment_cfg_loaded = extract_models_config(cfg_loaded)

    # Check if the models are the same
    return (
        image_cfg_used == image_cfg_loaded
        and caption_cfg_used == caption_cfg_loaded
        and alignment_cfg_used == alignment_cfg_loaded
    )