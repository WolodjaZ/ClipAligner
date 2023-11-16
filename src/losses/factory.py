from omegaconf import DictConfig, OmegaConf
from .base import BaseImageCaptionLoss, BaseImageLoss, BaseCaptionLoss, LossNotImplementedError
from .clip import ClipLoss, DistillClipLoss, SigLipLoss, CoCaLoss

def get_loss_fn(cfg: DictConfig | str) -> BaseImageCaptionLoss | BaseImageLoss | BaseCaptionLoss:
    """Get the loss function.
    
    Raises:
        LossNotImplementedError: If the loss function is not implemented.
    
    Args:
        cfg (DictConfig | str): Configuration file containing "name" key and other parameters.
            If str, then it should be the name of the loss function. Other parameters are set to default.
    Returns:
        BaseImageCaptionLoss | BaseImageLoss | BaseCaptionLoss: Loss function.
    """
    if isinstance(cfg, str):
        name = cfg.lower()
        dict_cfg = {}
    else:
        dict_cfg = OmegaConf.to_container(cfg, resolve=True)
        name = dict_cfg.pop("name").lower()
    if name == "clip":
        return ClipLoss(**dict_cfg)
    elif name == "coca":
        return CoCaLoss(**dict_cfg)
    elif name == "distillclip":
        return DistillClipLoss(**dict_cfg)
    elif name == "siglip":
        return SigLipLoss(**dict_cfg)
    else:
        raise LossNotImplementedError(f"Loss {name} is not implemented.")
