import torch
from src.utils import set_logger, get_loss_fn, set_seed, AverageMeter

def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    calculate_metrics: list | None = None) -> list:
    """Evaluate the model and calculate loss and metrics.
    
    Args:
        model (torch.nn.Module): Model.
        dataloader (torch.utils.data.DataLoader): Dataloader.
        loss_fn (torch.nn.Module): Loss function.
        device (torch.device): Device.
        calculate_metrics (list | None): List of metrics to calculate additional to the loss.
    Returns:
        list: List of metrics.
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
            images, captions = images.to(device), captions.to(device)
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

def evaluate():
    pass

def main():
    pass

if __name__ == "__main__":
    main()