"""Utility functions for Vietnamese ABSA system."""

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Union[Dict, List], file_path: Union[str, Path], indent: int = 2):
    """Save data to JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(file_path: Union[str, Path]) -> Union[Dict, List]:
    """Load data from JSON file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def count_parameters(model) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Convert seconds to human-readable time format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def get_device() -> torch.device:
    """Get available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_training_info(config: Dict[str, Any], model, train_dataset, val_dataset):
    """Print training configuration and dataset info."""
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Model: {config['model']['backbone']}")
    print(f"Max length: {config['model']['max_length']}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"\nEpochs: {config['training']['num_train_epochs']}")
    print(f"Batch size: {config['training']['per_device_train_batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Weight decay: {config['training']['weight_decay']}")
    print(f"FP16: {config['training']['fp16']}")
    print(f"Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    print("="*80 + "\n")


def ensure_dir(path: Union[str, Path]):
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    """Get current timestamp string."""
    return time.strftime("%Y%m%d_%H%M%S")


def merge_subword_labels(tokens: List[str], labels: List[str]) -> List[str]:
    """Merge subword tokens and their labels."""
    merged_tokens = []
    merged_labels = []
    
    current_token = ""
    current_label = "O"
    
    for token, label in zip(tokens, labels):
        if token.startswith("##"):
            current_token += token[2:]
        else:
            if current_token:
                merged_tokens.append(current_token)
                merged_labels.append(current_label)
            current_token = token
            current_label = label
    
    if current_token:
        merged_tokens.append(current_token)
        merged_labels.append(current_label)
    
    return merged_tokens, merged_labels


def calculate_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets."""
    from collections import Counter
    
    label_counts = Counter(labels)
    total = len(labels)
    
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weight = total / (num_classes * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)
