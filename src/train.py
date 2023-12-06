import os
import time
import torch
import hydra
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from omegaconf import DictConfig
from typing import Callable

from src.models import create_model, BaseImageCaptionModel, BaseImageModel, BaseCaptionModel
from src.datasets import get_dataset
from src.utils import set_logger, set_seed, get_custom_scheduler, AverageMeter, wrap_output
from src.losses import get_loss_fn, BaseImageCaptionLoss, BaseImageLoss, BaseCaptionLoss
from src.eval import validate

load_dotenv()
MAIN_CONFIG = os.getenv("MAIN_CONFIG")


def train_on_epoch(
    model: BaseImageCaptionModel | BaseImageModel | BaseCaptionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | Callable | None,
    loss_fn: BaseImageCaptionLoss | BaseImageLoss | BaseCaptionLoss,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    batch_freq_print: int = 100) -> (float, float | None):
    """Train the model on an epoch.
    
    Args:
        model (BaseImageCaptionModel | BaseImageModel | BaseCaptionModel): The neural network model.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scheduler (torch.optim.lr_scheduler._LRScheduler | None): Learning rate scheduler.
        loss_fn (BaseImageCaptionLoss | BaseImageLoss | BaseCaptionLoss): Loss function used for training.
        dataloader (torch.utils.data.DataLoader): DataLoader providing training data batches.
        device (torch.device): Device on which to perform computations.
        epoch (int): Current epoch.
        batch_freq_print (int): Frequency of logging training progress.
    Returns:
        float, float | None: Average loss and current learning rate. None if no scheduler is used.
    """
    # Set the model to train mode
    model.train()
    
    # Set the average meter
    loss_meter = AverageMeter()
    
    # Iterate over the dataloader
    for batch_idx, (images, captions) in enumerate(progress_bar := tqdm(dataloader)):
        # Move the data to the device
        images, captions = images.to(device), captions.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(images, captions)
        
        # wrap the output in a dictionary
        if not isinstance(output, dict):
            output = wrap_output(output)
        
        # Compute the loss
        losses = loss_fn(**output, output_dict=True)
        total_loss = sum(losses.values())
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Update the loss meter
        loss_meter.update(total_loss.item())
        
        # Log the loss
        if batch_idx % batch_freq_print == 0 or batch_idx == len(dataloader) - 1:
            progress_bar.set_description(f"Training - Batch {batch_idx}/{len(dataloader)} - Loss: {total_loss.item():.4f}")
    
    # Step the scheduler
    lr = optimizer.param_groups[0]["lr"]
    if scheduler:
        try:
            scheduler.step()
        except AttributeError:
            scheduler(epoch)
    
    return loss_meter.avg, lr
    
def train(cfg: DictConfig) -> None:
    """Main training function.
    
    Args:
        cfg (DictConfig): Configuration file.
    """
    # Set the seed
    set_seed(cfg.seed)
    logger.info(f"Seed: {cfg.seed}")

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Set the model
    model = create_model(cfg.model)
    logger.info(f"Model: {model}")

    # Set the dataset
    train_dataset, val_dataset = get_dataset(cfg.dataset, transfor=model.get_transformations())
    logger.info(f"Train Dataset: {train_dataset} | Validation Dataset: {val_dataset}")

    # Set the dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    if val_dataset is None:
        logger.info("No validation dataset is used.")
        val_dataloader = None
    else:
        val_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        )

    # Set the optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    
    # Load the checkpoint
    start_epoch = 0
    if cfg.checkpoint is not None:
        try:
            checkpoint = torch.load(cfg.checkpoint, map_location=device) # sourcery skip: extract-method
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            loss = checkpoint["loss"]
            logger.info(f"Checkpoint: {cfg.checkpoint} | Completed Epoch: {start_epoch} | Loss: {loss}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {cfg.checkpoint} so continuing without it | Error: {e}")

    # Set the scheduler
    if cfg.scheduler is None:
        scheduler = None
        logger.info("No scheduler is used.")
    else:
        if "_target_" not in cfg.scheduler:
            scheduler = get_custom_scheduler(cfg.scheduler, optimizer=optimizer)
        else:
            scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
        logger.info(f"Scheduler: {scheduler}")

    # Get the loss function
    loss_fn = get_loss_fn(cfg.loss)
    logger.info(f"Loss: {loss_fn}")

    # Set the output directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    save_dir = output_dir / cfg.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Save directory: {save_dir}")
    
    # Start training
    metrics = None
    logger.info(f"Start training for {cfg.epochs} epochs from {start_epoch} epoch...")
    for epoch in range(start_epoch, cfg.epochs):
        loss, lr  = train_on_epoch(model, optimizer, scheduler, loss_fn, train_dataloader, device, epoch, batch_freq_print=cfg.batch_freq_print)
        if val_dataloader is not None:
            metrics = validate(dataloader=val_dataloader, model=model, loss_fn=loss_fn, device=device)
        logger.info(f"Epoch: {epoch} | Loss: {loss:.4f} | LR: {lr} | Evaluate Metrics: {metrics}")
        if epoch % cfg.save_period == 0:
            checkpoint_dict = {
                "epoch": epoch,
                "loss": loss,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint_dict, save_dir / f"epoch_{epoch}.pt")
    

@hydra.main(version_base=None, config_name="train", config_path=MAIN_CONFIG)
def main(cfg: DictConfig) -> int:
    """Main function to train the model.
    
    Args:
        cfg (DictConfig): Configuration file.
    Returns:
        int: 0 if the training is successful. 1 otherwise.
    """
    # Set the logger
    set_logger(cfg.log_file, cfg.log_level, cfg.verbose)
    
    # start training
    start_time = time.time()
    try:
        train(cfg)
    except Exception as e:
        logger.exception(e)
        return 1
    
    # return 0 if the training is successful
    logger.info(f"Training FINISHED. Total time: {(time.time() - start_time) / 60:.2f} minutes")
    return 0
   

if __name__ == "__main__":
    main()