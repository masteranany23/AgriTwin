"""
Utility helper functions.
"""
import logging
import pickle
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def save_pickle(obj: Any, path: Path):
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    
    logger.info(f"Saved object to {path}")


def load_pickle(path: Path) -> Any:
    """
    Load object from pickle file.
    
    Args:
        path: Path to pickle file.
        
    Returns:
        Loaded object.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, "rb") as f:
        obj = pickle.load(f)
    
    logger.info(f"Loaded object from {path}")
    return obj


def setup_logging(log_level: str = "INFO"):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
