import os
import random
import numpy as np
import torch
import yaml


def seed_everything(seed=42):
    """
    Set random seeds across Python, NumPy, and PyTorch for deterministic and reproducible runs.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
