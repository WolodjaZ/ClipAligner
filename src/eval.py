import torch
import argparse
import lightning as L
from loguru import logger

from src.models import create_model
from src.datasets import get_dataset
from src.utils import parse_args, set_logger, set_seed, AverageMeter#, print_metrics
from src.losses import get_loss_fn

def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    fabric: L.fabric.Fabric,
    calculate_metrics: list | None = None) -> list:
    """Evaluate the model and calculate loss and metrics.
    
    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader providing training data batches.
        loss_fn (torch.nn.Module): Loss function used for training.
        fabric (L.fabric.Fabric): Lightning fabric.
        batch_idx_to_log (int): Frequency of logging training progress.
        calculate_metrics (list | None): List of additional metrics to calculate.
        
    Returns:
        List (list) of calculated metric values.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize the average meter
    loss_meter = AverageMeter()
    if calculate_metrics is not None and len(calculate_metrics) > 0:
        metrics_meter = {metric: AverageMeter() for metric in calculate_metrics}
    else:
        metrics_meter = {}
    
    # Iterate over the dataloader with torch.inference_mode()
    with torch.inference_mode():
        for images, captions in dataloader:
            # Move the data to the device
            images, captions = images.to(fabric.device), captions.to(fabric.device)
            # Forward pass
            image_embeddings, caption_embeddings = model(images, captions)
            # Compute the loss and update the loss meter
            loss = loss_fn(image_embeddings, caption_embeddings)
            loss_meter.update(loss.item())
            # Compute the metrics and update the metrics meter
            for metric_name, metric_meter in metrics_meter.items():
                metric = get_metric_value(metric_name, image_embeddings, caption_embeddings)
                metric_meter.update(metric)
    
    # Add the loss to the metrics meter
    metrics_meter['loss'] = loss_meter
    # Return the metrics
    return [metric_meter.avg for metric_meter in metrics_meter.values()]


def get_metric_value(metric_name: str, image_embeddings: torch.Tensor, caption_embeddings: torch.Tensor) -> float:
    """Get the metric value.
    
    Currently only supports:
        - cosine_similarity
    
    Raises:
        ValueError: If the metric is not supported.
    Args:
        metric_name (str): Metric name.
        image_embeddings (torch.Tensor): Image embeddings.
        caption_embeddings (torch.Tensor): Caption embeddings.
    Returns:
        float: Metric value.
    """
    if metric_name == 'cosine_similarity':
        return torch.nn.functional.cosine_similarity(image_embeddings, caption_embeddings).mean().item()
    else:
        raise ValueError(f"Metric {metric_name} is not supported.")


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate the model.
    
    Args:
        args (dict): Arguments.
        logger (logger): Logger.
    """
    # Set the seed
    set_seed(args.seed)
    logger.info(f"Seed: {args.seed}")
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    logger.info(f"Device: {device}")
    
    # Load the model
    model = create_model(args.model_name, model_checkpoint=args.model_checkpoint)
    model.to(device)
    logger.info(f"Model: {model}")
    
    # Get the dataset
    dataset = get_dataset(args.dataset_name, args.dataset_path)
    logger.info(f"Dataset: {dataset}")
    
    # Get the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logger.info("Initialized dataloader.")
    
    # Load the loss function for the metric
    loss_fn = get_loss_fn(args.loss_fn_name)
    
    # Evaluate
    logger.info("Start evaluation...")
    metrics = validate(model, dataloader, loss_fn, device, args.calculate_metrics)
    
    # Log the metrics
    logger.info(f"Metrics: {print_metrics(metrics)}")
    logger.info("Evaluation finished.")

def main() -> int:
    """Main function.
    
    Returns:
        int: Return code. 0 if success, 1 if error.
    """
    # Parse the arguments
    args = parse_args()
    
    # Set the logger
    set_logger(args)
    
    # evaluate
    try:
        evaluate(args)
    except Exception as e:
        logger.error(e)
        return 1
    
    return 0
    

if __name__ == "__main__":
    main()