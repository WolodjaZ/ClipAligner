import random
import numpy as np
import torch
from typing import List, Any
from jsonargparse import ArgumentParser, Namespace
import torch.distributed as dist
from omegaconf import DictConfig

try:
    from lightning.fabric import Fabric
except ImportError:
    Fabric = None

class AverageMeter:
    """
    Computes and stores the average, current value, sum, and count.

    Attributes:
        val (float): The current value.
        avg (float): The running average.
        sum (float): The sum of all values.
        count (int): The count of all entries.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        Resets all internal attributes to their initial state.
        """
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update the meter with a new value.

        Args:
            val (float): The new value to add.
            n (int, optional): The number of times to add val. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0

    def __add__(self, other: 'AverageMeter') -> 'AverageMeter':
        """
        Add two AverageMeter objects.

        Args:
            other (AverageMeter): The other AverageMeter to add.

        Returns:
            AverageMeter: A new AverageMeter instance with the sum of values.
        """
        if not isinstance(other, AverageMeter):
            raise ValueError("Only AverageMeter instances can be added")

        result = AverageMeter()
        result.sum = self.sum + other.sum
        result.count = self.count + other.count
        result.avg = result.sum / result.count if result.count != 0 else 0.0
        # The value 'val' is not aggregated as it represents the most recent value in each instance
        return result

def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility in random processes.

    Args:
        seed (int): The seed value to use for random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def wrap_output(output: torch.Tensor | List[torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Wrap the output in a dictionary based on the structure and length of the output.
    
    Raises:
        ValueError: If the output structure is not supported or the type is incorrect.

    Args:
        output (torch.Tensor | List[torch.Tensor]): The output to wrap. Can be a single torch.Tensor or a list of torch.Tensors.

    Returns:
        dict[str, torch.Tensor]: The wrapped output.
    """
    if isinstance(output, torch.Tensor):
        return {"output": output}

    if not isinstance(output, (list, tuple)):
        raise ValueError(f"Output type {type(output)} not supported.")

    output_length = len(output)
    if output_length == 1:
        keys = ["output"]
    elif output_length == 2:
        keys = ["image_features", "text_features"]
    elif output_length == 3:
        keys = ["image_features", "text_features", "logit_scale"]
    elif output_length == 4:
        keys = ["image_features", "text_features", "logit_scale", "logit_bias"]
    elif output_length == 5:
        keys = ["image_features", "text_features", "logits", "labels", "logit_scale"]
    elif output_length == 6:
        keys = ["image_features", "text_features", "logit_scale", "dist_image_features", "dist_logit_scale", "logit_bias"]
    else:
        raise ValueError(f"Output length {output_length} not supported.")

    return dict(zip(keys, output))


def all_gather_object(obj: Any, world_size: int = 1, fabric: Fabric | None=None) -> Any:
    """Gather objects from all processes.

    Args:
        obj (Any): Object to gather.
        world_size (int, optional): World size. Defaults to 1.
        fabric (Fabric | None, optional): Lightning fabric. Defaults to None.

    Returns:
        Any: The gathered objects.
    """
    if fabric is not None:
        return fabric.all_gather(obj)
    objects = [None for _ in range(world_size)]
    dist.all_gather_object(objects, obj)
    return objects


def parse_args() -> Namespace:
    """
    Parse command line arguments.

    Returns:
        Namespace: The parsed arguments.
    """
    parser = ArgumentParser("ClipAligner evaluation script")
    # Define your arguments here, for example:
    # parser.add_argument('--arg_name', type=int, default=42, help='Description of arg_name')
    return parser.parse_args()
