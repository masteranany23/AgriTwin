"""
Configuration loader for AgriTwin Bias Correction API.
"""
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class Config(BaseModel):
    """Complete configuration schema."""
    
    model: Dict[str, Any]
    data: Dict[str, Any]
    monitoring: Dict[str, Any]
    api: Dict[str, Any]
    system: Dict[str, Any]


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def load_config(env: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file based on environment.
    
    Args:
        env: Environment name (development, production). If None, uses ENV variable.
        
    Returns:
        Config: Loaded configuration object.
        
    Raises:
        FileNotFoundError: If config file not found.
        ValueError: If config is invalid.
    """
    if env is None:
        env = os.getenv("ENV", "development")
    
    config_path = get_project_root() / "config" / f"{env}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        config = Config(**config_dict)
        logger.info(f"Configuration loaded successfully (env={env})")
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {e}")


def get_model_path(config: Config, model_name: str = "ensemble") -> Path:
    """Get absolute path to model file."""
    root = get_project_root()
    return root / config.model["path"]


def get_gp_model_path(config: Config) -> Path:
    """Get absolute path to GP model file."""
    root = get_project_root()
    return root / config.model["gp_path"]


def get_data_path(config: Config) -> Path:
    """Get absolute path to training data."""
    root = get_project_root()
    return root / config.data["training_data_path"]


def get_ics_path(config: Config) -> Path:
    """Get absolute path to ICS ratios directory."""
    root = get_project_root()
    return root / config.data["ics_path"]


def get_log_dir(config: Config) -> Path:
    """Get absolute path to logs directory."""
    root = get_project_root()
    log_dir = root / config.monitoring["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
