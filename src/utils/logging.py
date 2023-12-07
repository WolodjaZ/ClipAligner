import sys
import hydra
from pathlib import Path
from loguru import logger


def set_logger(log_file: str, level: str, verbose: bool = False, rank: int = 0) -> None:
    """Set up a logger with specified settings.
    
    Raises:
        ValueError: If the logging level is invalid.
        OSError: If there is an error creating the log directory.
        Exception: If there is an error setting up the logger.

    Args:
        log_file (str): Path to the log file where logs will be written.
        level (str): Logging level, e.g., 'INFO', 'DEBUG', 'ERROR'.
        verbose (bool, optional): If True, also print logs to stdout. Defaults to False.
        rank (int, optional): Rank of the current process. Defaults to 0.
    
    Example usage:
        set_logger("path/to/logfile.log", "INFO", verbose=True)
    """
    # Validate logging level
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if level.upper() not in valid_levels:
        raise ValueError(f"Invalid logging level: {level}")

    # Get the log file path
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log_file = output_dir / log_file
    log_file = log_file if log_file.suffix else log_file / "file.log"
    
    # Create log directory if it doesn't exist
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # sourcery skip: raise-specific-error
        raise OSError(f"Error creating log directory: {e}") from e


    # Setup logger
    try:
        logger.remove() # Remove this line if you want to use multiple loggers
        logger.add(log_file, rotation="10 MB", level=level, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
        if verbose and rank == 0:
            logger.add(sys.stdout, colorize=True, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <blue>{level}</blue> | <level>{message}</level>")
    except Exception as e:
        # sourcery skip: raise-specific-error
        raise Exception(f"Error setting up logger: {e}") from e
