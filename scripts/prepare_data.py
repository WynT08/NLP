#!/usr/bin/env python3
"""Prepare raw data for Vietnamese ABSA."""

import argparse
import json
from pathlib import Path
import sys


def prepare_data(input_file: str, output_file: str):
    """
    Prepare raw data for ABSA training.
    
    Args:
        input_file: Input JSON file path
        output_file: Output JSON file path
    """
    print(f"Loading data from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Validate and standardize format
    prepared_data = []
    for i, item in enumerate(data):
        if 'text' not in item:
            print(f"Warning: Sample {i} missing 'text' field, skipping")
            continue
        
        # Ensure required fields
        prepared_item = {
            'text': item['text'],
            'aspects': item.get('aspects', []),
            'overall_sentiment': item.get('overall_sentiment', 'neutral')
        }
        
        # Validate aspects
        for aspect in prepared_item['aspects']:
            if 'term' not in aspect:
                print(f"Warning: Sample {i} has aspect without 'term' field")
            if 'sentiment' not in aspect:
                aspect['sentiment'] = 'neutral'
            if 'span' not in aspect and 'term' in aspect:
                # Try to find span
                term = aspect['term']
                text = prepared_item['text']
                start = text.find(term)
                if start != -1:
                    aspect['span'] = [start, start + len(term)]
                else:
                    aspect['span'] = [0, 0]
        
        prepared_data.append(prepared_item)
    
    # Save prepared data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prepared_data, f, ensure_ascii=False, indent=2)
    
    print(f"Prepared {len(prepared_data)} samples saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Prepare raw data for Vietnamese ABSA')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    prepare_data(args.input, args.output)


if __name__ == '__main__':
    main()
