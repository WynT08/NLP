#!/usr/bin/env python3
"""Validate lexicon coverage on dataset."""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lexicon_validator import validate_lexicon


def main():
    parser = argparse.ArgumentParser(description='Validate lexicon coverage')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file')
    parser.add_argument('--data', type=str, required=True, help='Dataset file to validate')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    parser.add_argument('--output', type=str, help='Output path for visualization')
    
    args = parser.parse_args()
    
    # Validate lexicon
    coverage_stats = validate_lexicon(args.config, args.data)
    
    # Optionally visualize
    if args.visualize:
        from src.lexicon_validator import LexiconValidator
        from src.utils import load_config
        
        config = load_config(args.config)
        validator = LexiconValidator(config)
        
        output_path = args.output or 'lexicon_coverage.png'
        validator.visualize_coverage(coverage_stats, save_path=output_path)
        print(f"Visualization saved to {output_path}")


if __name__ == '__main__':
    main()
