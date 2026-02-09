#!/usr/bin/env python3
"""Augment training data for Vietnamese ABSA."""

import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_augmentation import create_augmenter
from src.utils import set_seed


def augment_data(
    input_file: str,
    output_file: str,
    config_path: str = "config/config.yaml",
    augmentation_factor: float = 1.3,
    seed: int = 42
):
    """
    Augment training data.
    
    Args:
        input_file: Input JSON file (training data)
        output_file: Output JSON file (augmented data)
        config_path: Path to configuration file
        augmentation_factor: Target size multiplier
        seed: Random seed
    """
    set_seed(seed)
    
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_size = len(data)
    target_size = int(original_size * augmentation_factor)
    
    print(f"Original dataset size: {original_size}")
    print(f"Target size: {target_size} (factor: {augmentation_factor}x)")
    
    print("\nInitializing augmenter...")
    augmenter = create_augmenter(config_path)
    
    print("\nAugmenting data...")
    augmented_data = augmenter.augment_dataset(
        data,
        target_size=target_size,
        augmentation_factor=augmentation_factor
    )
    
    # Save augmented data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nAugmented data saved to {output_file}")
    print(f"Final dataset size: {len(augmented_data)}")
    print(f"Added {len(augmented_data) - original_size} new samples")


def main():
    parser = argparse.ArgumentParser(description='Augment Vietnamese ABSA training data')
    parser.add_argument('--input', type=str, required=True, help='Input training JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output augmented JSON file')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file')
    parser.add_argument('--factor', type=float, default=1.3, help='Augmentation factor (default: 1.3)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    augment_data(args.input, args.output, args.config, args.factor, args.seed)


if __name__ == '__main__':
    main()
