## Custom Scheduler for learning rate from open_clip repository
## Implementation from https://github.com/mlfoundations/open_clip/blob/main/src/training/scheduler.py
## License: MIT; https://github.com/mlfoundations/open_clip/blob/main/LICENSE


import numpy as np
from torch.optim.optimizer import Optimizer
from omegaconf import DictConfig, OmegaConf
from typing import Callable


class CustomSchedulerNotImplementedError(NotImplementedError):
    """Raised when the custom scheduler is not implemented."""
    pass


def assign_learning_rate(optimizer: Optimizer, new_lr: float) -> None:
    """
    Assign a new learning rate to an optimizer.

    Args:
        optimizer (Optimizer): The optimizer to update.
        new_lr (float): The new learning rate value to assign.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr: float, warmup_length: int, step: int) -> float:
    """
    Calculate the learning rate for the warmup phase.

    Args:
        base_lr (float): The base learning rate after warmup.
        warmup_length (int): The number of steps for warmup.
        step (int): The current step.

    Returns:
        float: The calculated learning rate.
    """
    return base_lr * (step + 1) / warmup_length

def const_lr(optimizer: Optimizer, base_lr: float, warmup_length: int, steps: int) -> Callable[[int], float]:
    """
    Learning rate schedule with a constant learning rate after warmup.

    Args:
        optimizer (Optimizer): The optimizer to adjust.
        base_lr (float): The base learning rate.
        warmup_length (int): The number of steps for warmup.
        steps (int): The total number of steps.

    Returns:
        Callable[[int], float]: A function to adjust learning rate at each step.
    """
    def _lr_adjuster(step: int) -> float:
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def const_lr_cooldown(optimizer: Optimizer, base_lr: float, warmup_length: int, steps: int, cooldown_steps: int, 
                      cooldown_power: float = 1.0, cooldown_end_lr: float = 0.0) -> Callable[[int], float]:
    """
    Learning rate schedule with a constant learning rate and a cooldown phase.

    Args:
        optimizer (Optimizer): The optimizer to adjust.
        base_lr (float): The base learning rate.
        warmup_length (int): The number of steps for warmup.
        steps (int): The total number of steps.
        cooldown_steps (int): The number of steps for cooldown.
        cooldown_power (float): The power for polynomial cooldown. Defaults to 1.0 (linear).
        cooldown_end_lr (float): The ending learning rate after cooldown. Defaults to 0.0.

    Returns:
        Callable[[int], float]: A function to adjust learning rate at each step.
    """
    def _lr_adjuster(step: int) -> float:
        start_cooldown_step = steps - cooldown_steps
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if step < start_cooldown_step:
                lr = base_lr
            else:
                e = step - start_cooldown_step
                es = steps - start_cooldown_step
                decay = (1 - (e / es)) ** cooldown_power
                lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def cosine_lr(optimizer: Optimizer, base_lr: float, warmup_length: int, steps: int) -> Callable[[int], float]:
    """
    Learning rate schedule with a cosine decay after warmup.

    Args:
        optimizer (Optimizer): The optimizer to adjust.
        base_lr (float): The base learning rate.
        warmup_length (int): The number of steps for warmup.
        steps (int): The total number of steps.

    Returns:
        Callable[[int], float]: A function to adjust learning rate at each step.
    """
    def _lr_adjuster(step: int) -> float:
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def get_custom_scheduler(cfg: dict | DictConfig, optimizer: Optimizer) -> Callable[[int], float]:
    """Get the custom scheduler.
    
    Raises:
        CustomSchedulerNotImplementedError: If the custom scheduler is not implemented.
    
    Args:
        cfg (dict | DictConfig): Configuration file containing "name" key and other parameters.
        optimizer (Optimizer): The optimizer to adjust.
    Returns:
        Callable[[int], float]: A function to adjust learning rate at each step.
    """
    # Transform the cfg to dictionary
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    
    # Extract the name and get the dictionary of parameters
    name = cfg.pop("name", None)
    if name is None:
        raise CustomSchedulerNotImplementedError("Custom Scheduler name is not provided.")
    
    name = name.lower()
    # Get the loss function based on the name
    if name == "const_lr":
        return const_lr(optimizer, **cfg)
    elif name == "const_lr_cooldown":
        return const_lr_cooldown(optimizer, **cfg)
    elif name == "cosine_lr":
        return cosine_lr(optimizer, **cfg)
    else:
        raise CustomSchedulerNotImplementedError(f"Custom Scheduler {name} is not implemented.")