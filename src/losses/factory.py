from omegaconf import DictConfig, OmegaConf
from .base import BaseImageCaptionLoss, BaseImageLoss, BaseCaptionLoss, LossNotImplementedError
from .clip import ClipLoss, DistillClipLoss, SigLipLoss, CoCaLoss

def get_loss_fn(cfg: dict | DictConfig) -> BaseImageCaptionLoss | BaseImageLoss | BaseCaptionLoss:
    """Get the loss function.
    
    Raises:
        LossNotImplementedError: If the loss function is not implemented.
    
    Args:
        cfg (dict | DictConfig): Configuration file containing "name" key and other parameters.
    Returns:
        BaseImageCaptionLoss | BaseImageLoss | BaseCaptionLoss: Loss function.
    """
    # Transform the cfg to dictionary
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    
    # Extract the name and get the dictionary of parameters
    name = cfg.pop("name", None)
    if name is None:
        raise LossNotImplementedError("Loss name is not provided.")

    name = name.lower()
    # Get the loss function based on the name
    if name == "clip":
        return ClipLoss(**cfg)
    elif name == "coca":
        return CoCaLoss(**cfg)
    elif name == "distillclip":
        return DistillClipLoss(**cfg)
    elif name == "siglip":
        return SigLipLoss(**cfg)
    else:
        raise LossNotImplementedError(f"Loss {name} is not implemented.")
