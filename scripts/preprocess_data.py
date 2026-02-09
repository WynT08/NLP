#!/usr/bin/env python3
"""Preprocess data for Vietnamese ABSA."""

import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import create_preprocessor
from src.utils import load_config


def preprocess_data(
    input_file: str,
    output_file: str,
    config_path: str = "config/config.yaml"
):
    """
    Preprocess data using VietnamesePreprocessor.
    
    Args:
        input_file: Input JSON file
        output_file: Output JSON file
        config_path: Path to configuration file
    """
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    
    print("Initializing preprocessor...")
    preprocessor = create_preprocessor(config_path)
    
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Preprocessing {len(data)} samples...")
    preprocessed_data = []
    
    for i, item in enumerate(data):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(data)} samples")
        
        text = item['text']
        
        # Preprocess with masks
        result = preprocessor.preprocess(text, return_masks=True)
        
        # Create preprocessed item
        preprocessed_item = {
            'text': item['text'],  # Keep original
            'processed_text': result['processed_text'],
            'aspects': item.get('aspects', []),
            'overall_sentiment': item.get('overall_sentiment', 'neutral'),
            'negation_mask': result.get('negation_mask', None),
            'intensity_mask': result.get('intensity_mask', None),
            'emojis': result.get('emojis', [])
        }
        
        preprocessed_data.append(preprocessed_item)
    
    # Save preprocessed data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(preprocessed_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nPreprocessed {len(preprocessed_data)} samples saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess Vietnamese ABSA data')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output, args.config)


if __name__ == '__main__':
    main()
