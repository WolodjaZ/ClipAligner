import os
import time
import torch
import hydra
import rootutils
import lightning as L
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import DictConfig, open_dict
from lightning.fabric.loggers import TensorBoardLogger
from typing import Callable

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True, cwd=False)

from src.models import create_model, BaseImageCaptionModel, BaseImageModel, BaseCaptionModel
from src.datasets import get_dataset
from src.utils import set_logger, set_seed, get_custom_scheduler, AverageMeter, wrap_output, format_dict_print
from src.losses import get_loss_fn, BaseImageCaptionLoss, BaseImageLoss, BaseCaptionLoss
from src.eval import validate

MAIN_CONFIG = os.getenv("MAIN_CONFIG")


def train_on_epoch(
    model: BaseImageCaptionModel | BaseImageModel | BaseCaptionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | Callable | None,
    loss_fn: BaseImageCaptionLoss | BaseImageLoss | BaseCaptionLoss,
    dataloader: torch.utils.data.DataLoader,
    fabric: L.fabric.Fabric,
    epoch: int,
    batch_freq_print: int = 100) -> (float, float | None):
    """Train the model on an epoch.
    
    Args:
        model (BaseImageCaptionModel | BaseImageModel | BaseCaptionModel): The neural network model.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scheduler (torch.optim.lr_scheduler._LRScheduler | None): Learning rate scheduler.
        loss_fn (BaseImageCaptionLoss | BaseImageLoss | BaseCaptionLoss): Loss function used for training.
        dataloader (torch.utils.data.DataLoader): DataLoader providing training data batches.
        fabric (L.fabric.Fabric): Lightning fabric.
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
    for batch_idx, (images, captions) in enumerate(progress_bar := tqdm(dataloader, disable=fabric.global_rank != 0)):
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
        fabric.backward(total_loss)
        optimizer.step()
        
        # Update the loss meter
        loss_meter.update(total_loss.item())
        
        # Log the loss
        if batch_idx % batch_freq_print == 0 or batch_idx == len(dataloader) - 1:
            progress_bar.set_description(
                f"Train Epoch: {epoch} ({int(100.0 * (batch_idx+1) / len(dataloader)):2d}%) Loss: {total_loss.item():.6f}"
            )
    
    # Step the scheduler
    lr = optimizer.param_groups[0]["lr"]
    if scheduler:
        try:
            scheduler.step()
        except AttributeError:
            scheduler(epoch)
    
    return loss_meter.avg, lr
    
def train(cfg: DictConfig, fabric: L.fabric.Fabric, output_dir: Path ) -> None:
    """Main training function.
    
    Args:
        cfg (DictConfig): Configuration file.
        fabric (L.fabric.Fabric): Lightning fabric.
        output_dir (Path): Path to the output directory.
    """
    # Set the seed
    fabric.seed_everything(cfg.seed)
    # set_seed(cfg.seed)
    logger.info(f"Seed: {cfg.seed}")

    # Set the device
    logger.info(f"Device: {fabric.device}")

    # Set the model
    model = create_model(cfg.model)
    logger.info(f"Model: {model}")

    # Set the dataset. Let rank 0 download the data first, then everyone will load MNIST
    with fabric.rank_zero_first(local=False):  # set `local=True` if your filesystem is not shared between machines
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
    
    # Set the model and optimizer
    model = fabric.setup(model)
    optimizer = fabric.setup_optimizers(optimizer)
    
    # compile torch model, thread about it: https://github.com/Lightning-AI/pytorch-lightning/issues/17250
    if cfg.compile:
        logger.info("Compiling the model with torch.compile ...")
        torch.compile(model)
    
    # Set the dataloaders
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    if val_dataloader is not None:
        val_dataloader = fabric.setup_dataloaders(val_dataloader)
    
    # Load the checkpoint
    start_epoch = 0
    if cfg.checkpoint is not None:
        try:
            
            checkpoint = fabric.load(cfg.checkpoint) # sourcery skip: extract-method
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
    with open_dict(cfg):
        cfg.loss.rank = cfg.global_rank
        cfg.loss.world_size = cfg.world_size
    loss_fn = get_loss_fn(cfg.loss)
    logger.info(f"Loss: {loss_fn}")

    # Set the output directory
    save_dir = output_dir / cfg.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Save directory: {save_dir}")
    
    # WAit for distributed nodes to finish
    if cfg.world_size > 1:
        logger.info("Waiting for distributed nodes to finish setting up...")
        fabric.barrier("setup")
    
    # Start training
    metrics = None
    logger.info(f"Start training for {cfg.epochs} epochs from {start_epoch} epoch...")
    for epoch in range(start_epoch, cfg.epochs):
        loss, lr  = train_on_epoch(model, optimizer, scheduler, loss_fn, train_dataloader, fabric, epoch, batch_freq_print=cfg.batch_freq_print)
        fabric.log_dict({"train_loss": loss, "lr": lr})
        if val_dataloader is not None:
            metrics = validate(model=model, dataloader=val_dataloader, fabric=fabric, epoch=epoch, calculate_metrics=cfg.calculate_metrics, batch_freq_print=cfg.batch_freq_print)
            fabric.log_dict(metrics)
            
        logger.info(f"Epoch: {epoch} | Loss: {loss:.4f} | LR: {lr} | Evaluate Metrics:{format_dict_print(metrics)}")
        if epoch % cfg.save_period == 0:
            checkpoint_dict = {
                "epoch": epoch,
                "loss": loss,
                "state_dict": model,
                "optimizer": optimizer,
            }
            fabric.save(save_dir / f"epoch_{epoch}.ckpt", checkpoint_dict)
            logger.info(f"Checkpoint is saved at {save_dir / f'epoch_{epoch}.ckpt'}")

        # Wait for distributed nodes to finish
        if cfg.world_size > 1:
            fabric.barrier("epoch")
        
        # Log the memory usage
        if fabric.device.type == "cuda":
            logger.debug(torch.cuda.memory_summary())

@hydra.main(version_base=None, config_name="train", config_path=MAIN_CONFIG)
def main(cfg: DictConfig) -> int:
    """Main function to train the model.
    
    Args:
        cfg (DictConfig): Configuration file.
    Returns:
        int: 0 if the training is successful. 1 otherwise.
    """
    # # Set the GPU
    # if torch.cuda.is_available():
    #     # This enables tf32 on Ampere GPUs which is only 8% slower than
    #     # float16 and almost as accurate as float32
    #     # This was a default in pytorch until 1.12
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.benchmark = True
    #     torch.backends.cudnn.deterministic = False
    
    # Set the loggers
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    loggers = [TensorBoardLogger(root_dir=output_dir)] if cfg.vis_logger else None
    
    # Set the pytorch lightning
    fabric = L.Fabric(**cfg.fabric, loggers=loggers)
    fabric.launch()
    
    # Ad distributed params to the config
    with open_dict(cfg):
        cfg.global_rank = fabric.global_rank
        cfg.world_size = fabric.world_size
        cfg.local_rank = fabric.local_rank
        cfg.node_rank = fabric.node_rank
    
    # Set the logger
    set_logger(cfg.log_file, cfg.log_level, cfg.verbose, cfg.global_rank)
    logger.info(f"Fabric: {fabric} is launched.")
    if cfg.world_size > 1:
        logger.info(f"Running in distributed mode with {cfg.world_size} nodes.")
    
    # start training
    error_code = 0
    start_time = time.time()
    try:
        train(cfg, fabric=fabric, output_dir=output_dir)
    except Exception as e:
        logger.exception(e)
        error_code = 1
    
    # Log the total time and memory usage
    logger.info(f"Training FINISHED. Total time: {(time.time() - start_time) / 60:.2f} minutes")
    # return 0 if the training is successful
    return error_code
   

if __name__ == "__main__":
    main()