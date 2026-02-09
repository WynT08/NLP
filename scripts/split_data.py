#!/usr/bin/env python3
"""Split dataset into train/val/test sets."""

import argparse
import json
import random
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split


def split_data(
    input_file: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True
):
    """
    Split data into train/validation/test sets.
    
    Args:
        input_file: Input JSON file
        output_dir: Output directory for split files
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        stratify: Whether to use stratified split
    """
    random.seed(seed)
    
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_samples = len(data)
    print(f"Total samples: {total_samples}")
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    if stratify:
        # Stratify by overall sentiment
        labels = [item.get('overall_sentiment', 'neutral') for item in data]
        label_counts = Counter(labels)
        
        print(f"\nOverall sentiment distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({count/total_samples*100:.1f}%)")
        
        # First split: train vs (val + test)
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            data,
            labels,
            train_size=train_ratio,
            random_state=seed,
            stratify=labels
        )
        
        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data,
            train_size=val_ratio_adjusted,
            random_state=seed,
            stratify=temp_labels
        )
    else:
        # Random split
        random.shuffle(data)
        
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
    
    # Save splits
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        output_file = output_path / f'{split_name}.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n{split_name.capitalize()} set: {len(split_data)} samples")
        print(f"  Saved to {output_file}")
        
        # Print distribution
        if stratify:
            split_labels = [item.get('overall_sentiment', 'neutral') for item in split_data]
            split_label_counts = Counter(split_labels)
            for label, count in split_label_counts.items():
                print(f"    {label}: {count} ({count/len(split_data)*100:.1f}%)")
    
    print(f"\nSplit complete!")
    print(f"  Train: {len(train_data)} ({len(train_data)/total_samples*100:.1f}%)")
    print(f"  Val: {len(val_data)} ({len(val_data)/total_samples*100:.1f}%)")
    print(f"  Test: {len(test_data)} ({len(test_data)/total_samples*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Split Vietnamese ABSA dataset')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-stratify', action='store_true', help='Disable stratified split')
    
    args = parser.parse_args()
    
    split_data(
        args.input,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        not args.no_stratify
    )


if __name__ == '__main__':
    main()
