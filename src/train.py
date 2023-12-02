import os
import torch
import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from src.models import create_model
from src.datasets import get_dataset
from src.utils import set_logger, set_seed, AverageMeter
from src.losses import get_loss_fn
from src.eval import validate


def train_on_epoch(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    loss_fn: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    batch_idx_to_log: int = 100) -> (float, float | None):
    """Train the model on an epoch.
    
    Args:
        model (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scheduler (torch.optim.lr_scheduler._LRScheduler | None): Learning rate scheduler.
        loss_fn (torch.nn.Module): Loss function used for training.
        dataloader (torch.utils.data.DataLoader): DataLoader providing training data batches.
        device (torch.device): Device on which to perform computations.
        batch_idx_to_log (int): Frequency of logging training progress.
    Returns:
        float, float | None: Average loss and current learning rate. None if no scheduler is used.
    """
    # Set the model to train mode
    model.train()
    
    # Set the average meter
    loss_meter = AverageMeter()
    
    # Iterate over the dataloader
    for batch_idx, (images, captions) in enumerate(tqdm(dataloader, desc="Training")):
        # Move the data to the device
        images, captions = images.to(device), captions.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        image_embeddings, caption_embeddings = model(images, captions)
        
        # Compute the loss
        loss = loss_fn(image_embeddings, caption_embeddings)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update the loss meter
        loss_meter.update(loss.item())
        
        # Log the loss
        if batch_idx % batch_idx_to_log == 0 or batch_idx == len(dataloader) - 1:
            logger.info(f"Batch: {batch_idx} | Loss: {loss.item()}")
    
    # Step the scheduler
    lr = optimizer.param_groups[0]["lr"] if scheduler else None
    if scheduler:
        scheduler.step()
    
    return loss_meter.avg, lr
    
def train(cfg: DictConfig) -> None:
    """Main training function.
    
    Args:
        cfg (DictConfig): Configuration file.
    """
    # Set the seed
    set_seed(cfg.seed)
    logger.info(f"Seed: {cfg.train.seed}")
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Set the dataset
    train_dataset, val_dataset = get_dataset(**OmegaConf.to_container(cfg.dataset, resolve=True))
    logger.info(f"Train Dataset: {train_dataset} | Validation Dataset: {val_dataset}")
    
    # Set the dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True
    )
    if val_dataset is None:
        logger.info("No validation dataset is used.")
        val_dataloader = None
    else:
        val_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=True
        )
    
    # Set the model
    model = create_model(cfg.model)
    logger.info(f"Model: {model}")
    
    # Set the optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    
    # Set the scheduler
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    if scheduler is None:
        logger.info("No scheduler is used.")
    else:
        logger.info(f"Scheduler: {scheduler}")
    
    # Get the loss function
    loss_fn = get_loss_fn(cfg.loss)
    if loss_fn is None:
        raise NotImplementedError("Loss function is not implemented.")
    logger.info("Loss function initialized")
    
    # Set the evaluation function
    current_validator = (
        partial(validate, dataloader=val_dataloader, model=model, loss_fn=loss_fn, device=device)
        if val_dataloader is not None else
        partial(validate, dataloader=train_dataloader, model=model, loss_fn=loss_fn, device=device))
    
    # Start training
    logger.info(f"Start training for {cfg.train.epochs} epochs...")
    for epoch in range(cfg.train.epochs):
        loss, lr  = train_on_epoch(model, optimizer, scheduler, loss_fn, train_dataloader, device)
        metrics = current_validator()
        logger.info(f"Epoch: {epoch} | Loss: {loss:.4f} | LR: {lr} | Evaluate Metrics: {metrics}")
    

@hydra.main(version_base=None, config_name="config", config_path=os.environ["MAIN_CONFIG"])
def main(cfg: DictConfig) -> int:
    """Main function to train the model.
    
    Args:
        cfg (DictConfig): Configuration file.
    Returns:
        int: 0 if the training is successful. 1 otherwise.
    """
    # Set the logger
    set_logger(cfg)
    
    # start training
    try:
        train(cfg)
    except Exception as e:
        logger.exception(e)
        return 1
    
    # return 0 if the training is successful
    logger.info("Training FINISHED.")
    return 0
   

if __name__ == "__main__":
    main()