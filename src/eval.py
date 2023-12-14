import time
import torch
import rootutils
import lightning as L
from tqdm import tqdm
from loguru import logger
from jsonargparse import Namespace
import torch.nn.functional as F

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True, cwd=False)

from src.models import create_model, BaseImageCaptionModel, BaseImageModel, BaseCaptionModel
from src.datasets import get_dataset
from src.utils import parse_args, set_logger, AverageMeter, wrap_output, all_gather_object, format_dict_print


ALLOWED_METRICS = ["clip_loss", "generative_loss", "image_to_text", "text_to_image"]


def get_clip_metrics(image_embeddings, caption_embeddings, logit_scale):
    """
    Compute various similarity metrics between image and text embeddings.

    Args:
        image_embeddings (torch.Tensor): Embeddings of the images.
        caption_embeddings (torch.Tensor): Embeddings of the texts.
        logit_scale (torch.Tensor): Scale for logits.

    Returns:
        Dict[str, float]: A dictionary containing computed metrics.
    """
    # Calculate logits and detach from computation graph
    logits_per_image = (logit_scale * image_embeddings @ caption_embeddings.t()).detach().cpu()
    logits_per_text = logits_per_image.t()

    # Initialize metrics dictionary
    metrics = {}
    ground_truth = torch.arange(len(caption_embeddings)).to(logits_per_image.device)

    # Helper function to calculate metrics
    def calculate_metrics(logit: torch.Tensor, name: str) -> None:
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth.unsqueeze(1))[1]
        metrics[f"{name}_mean_rank"] = preds.float().mean().item() + 1
        metrics[f"{name}_median_rank"] = preds.median().item() + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = (preds < k).float().mean().item()

    # Calculate metrics for both directions
    calculate_metrics(logits_per_image, "image_to_text")
    calculate_metrics(logits_per_text, "text_to_image")

    return metrics


def maybe_compute_generative_loss(model_out: dict) -> torch.Tensor | None:
    """Compute the generative loss. For such models that are trained with coca loss.
    
    Args:
        model_out (dict): Model output.
    Returns:
        torch.Tensor | None: Generative loss.
    """
    # Check if logits and labels are in the model output
    if "logits" in model_out and "labels" in model_out:
        # Extract the logits and labels
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        # Compute the loss
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)

    return None


def validate(
    model: BaseImageCaptionModel | BaseImageModel | BaseCaptionModel,
    dataloader: torch.utils.data.DataLoader,
    fabric: L.fabric.Fabric,
    epoch: int,
    calculate_metrics: list | None = None) -> list:
    """Evaluate the model and calculate loss and metrics.
    
    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader providing training data batches.
        fabric (L.fabric.Fabric): Lightning fabric.
        epoch (int): Current epoch.
        batch_idx_to_log (int): Frequency of logging training progress.
        calculate_metrics (list | None): List of additional metrics to calculate. 
            If None all metrics are calculated. If empty list no metrics are calculated.
            Only supported metrics are allowed: ALLOWED_METRICS.
    Returns:
        List (list) of calculated metric values.
    """
    # Set the model to evaluation mode
    model.eval()

    #Validate the calculate_metrics
    calculate_metrics = ALLOWED_METRICS if calculate_metrics is None else calculate_metrics
    metrics_meter = {metric: AverageMeter() for metric in calculate_metrics if metric in ALLOWED_METRICS}
    
    # Return if no metrics are calculated
    if not metrics_meter:
        return {}

    # Iterate over the dataloader with torch.inference_mode()
    all_image_embeddings, all_text_embeddings = [], []
    with torch.inference_mode():
        all_image_embeddings, all_text_embeddings = [], []
        for batch_idx, (images, captions) in enumerate(progress_bar := tqdm(dataloader, disable=fabric.global_rank != 0)):
            output = model(images, captions)
            
            # wrap the output in a dictionary
            output_wrap = wrap_output(output)

            # extract the image and text embeddings
            image_embeddings = output_wrap.get('image_features', None)
            caption_embeddings = output_wrap.get('text_features', None)
            logit_scale = output_wrap.get('logit_scale', None)

            # Check if image_embeddings and caption_embeddings are None. For now we only support image-caption datasets.
            if image_embeddings is None or caption_embeddings is None:
                raise ValueError("Image or caption embeddings are missing. For now we only support image-caption datasets.")

            # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
            # however, system RAM is easily exceeded and compute time becomes problematic
            all_image_embeddings.append(image_embeddings.cpu())
            all_text_embeddings.append(caption_embeddings.cpu())
            logit_scale = logit_scale.mean()

            # compute logits and loss
            batch_size = images.shape[0]
            logits_per_image = logit_scale * image_embeddings @ caption_embeddings.t()
            logits_per_text = logits_per_image.t()
            labels = torch.arange(batch_size, device=fabric.device).long()
            total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
            
            # Update metrics
            gen_loss = maybe_compute_generative_loss(output_wrap)
            if "generative_loss" in calculate_metrics and gen_loss is not None:
                metrics_meter["generative"].update(gen_loss.item(), batch_size)
            if "clip_loss" in calculate_metrics:
                metrics_meter["clip_loss"].update(total_loss.item(), batch_size)

            # Log the loss
            progress_bar.set_description(
                f"Validating Epoch: {epoch} ({int(100.0 * (batch_idx+1) / len(dataloader)):2d}%) Loss: {total_loss.item():.6f}"
            )

        # Gather additional metrics
        additional_metrics = get_clip_metrics(
            image_embeddings=torch.cat(all_image_embeddings),
            caption_embeddings=torch.cat(all_text_embeddings),
            logit_scale=logit_scale.cpu(),
        )
        
        # Update the metrics meter
        for metric_name, metric_value in additional_metrics.items():
            if metric_name in metrics_meter:
                metrics_meter[metric_name].update(metric_value) #TODO: check #,len(dataloader.dataset))

    # all_gather is used to aggregated the value across processes
    metrics_meter = all_gather_object(metrics_meter, fabric=fabric)
    if isinstance(metrics_meter, list):
        # Average the metrics
        metrics_meter_keys = list(metrics_meter[0].keys())
        metrics_meter = {
            metric_name: sum(
                process_metric[metric_name] for process_metric in metrics_meter
            )
            for metric_name in metrics_meter_keys
        }
    
    # Return the metrics
    return {f"val_{key}": metric_meter.avg for key, metric_meter in metrics_meter.items()}


def evaluate(args: Namespace, fabric: L.fabric.Fabric) -> None:
    """Evaluate the model.
    
    Args:
        args (Namespace): Arguments.
        fabric (L.fabric.Fabric): Lightning fabric.
    """
    # Set the seed
    fabric.seed_everything(args.seed)
    logger.info(f"Seed: {args.seed}")
    
    # Set the device
    logger.info(f"Device: {fabric.device}")
    
    # Load the checkpoint
    try:
        checkpoint = fabric.load(args.checkpoint)
        model_params = checkpoint["params"]
        model_state_dict = checkpoint["state_dict"]
        logger.info(f"Checkpoint: {args.checkpoint} with params: {model_params}")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
    
    # Set the model
    model = create_model(model_params)
    logger.info(f"Model: {model}")
    
    # Get the dataset
    with fabric.rank_zero_first(local=False):  # set `local=True` if your filesystem is not shared between machines
        dataset, _ = get_dataset({"name": args.dataset,
                                  "val": {
                                            "dataset_path": args.dataset_path, 
                                            "text_max_length": args.text_max_length,
                                            "train_size": args.train_size}},
                                 transfor=model.get_transformations())
    logger.info(f"Dataset: {dataset}")
    
    # Set the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Set the model and dataloader
    dataloader = fabric.setup_dataloaders(dataloader)
    model = fabric.setup(model)
    if args.compile:
        logger.info("Compiling the model with torch.compile ...")
        torch.compile(model)

    model.load_state_dict(model_state_dict)
    
    # Wait for distributed nodes to finish
    if fabric.world_size > 1:
        logger.info("Waiting for distributed nodes to finish setting up...")
        fabric.barrier("setup")
    
    # Evaluate
    logger.info("Start evaluation...")
    metrics = validate(model=model, dataloader=dataloader, fabric=fabric, epoch=0, calculate_metrics=None)
    
    # Log the metrics
    logger.info(f"Metrics: {format_dict_print(metrics)}")


def main() -> int:
    """Main function.
    
    Returns:
        int: Return code. 0 if success, 1 if error.
    """
    # Parse the arguments
    args = parse_args()
    
    # Set the pytorch lightning
    fabric = L.Fabric(accelerator=args.accelerator, strategy=args.strategy)
    fabric.launch()
    
    # Set the logger
    set_logger(log_file="", level="INFO", verbose=True, rank=fabric.global_rank, train=False)
    logger.info(f"Fabric: {fabric} is launched.")
    if fabric.world_size > 1:
        logger.info(f"Running in distributed mode with {fabric.world_size} nodes.")
    
    # start evaluation
    error_code = 0
    start_time = time.time()
    try:
        evaluate(args, fabric=fabric)
    except Exception as e:
        logger.exception(e)
        error_code = 1
    
    # Log the total time and memory usage
    logger.info(f"Evaluation FINISHED. Total time: {(time.time() - start_time) / 60:.2f} minutes")
    # return 0 if the training is successful
    return error_code


if __name__ == "__main__":
    main()