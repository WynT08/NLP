#!/usr/bin/env python3
"""Evaluate ABSA pipeline on test data."""

import argparse
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import create_pipeline
from src.evaluation import ABSAEvaluator, save_evaluation_results
from src.utils import load_json


def evaluate(
    stage1_model_path: str,
    stage2_model_path: str,
    test_data_path: str,
    config_path: str = "config/config.yaml",
    output_path: str = None
):
    """
    Evaluate ABSA pipeline on test data.
    
    Args:
        stage1_model_path: Path to stage 1 model
        stage2_model_path: Path to stage 2 model
        test_data_path: Path to test data
        config_path: Path to configuration
        output_path: Path to save evaluation results
    """
    print("="*80)
    print("ABSA PIPELINE EVALUATION")
    print("="*80)
    
    # Create pipeline
    print("\nInitializing ABSA pipeline...")
    pipeline = create_pipeline(
        stage1_model_path=stage1_model_path,
        stage2_model_path=stage2_model_path,
        config_path=config_path
    )
    
    # Load test data
    print(f"\nLoading test data from {test_data_path}...")
    test_data = load_json(test_data_path)
    print(f"Loaded {len(test_data)} test samples")
    
    # Create evaluator
    evaluator = ABSAEvaluator(pipeline)
    
    # Evaluate
    print("\nEvaluating pipeline...")
    metrics, predictions = evaluator.evaluate_dataset(test_data)
    
    # Save results
    if output_path:
        save_evaluation_results(metrics, predictions, output_path)
    
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description='Evaluate Vietnamese ABSA pipeline')
    parser.add_argument('--stage1-model', type=str, required=True,
                       help='Path to Stage 1 model directory')
    parser.add_argument('--stage2-model', type=str, required=True,
                       help='Path to Stage 2 model directory')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data JSON file')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    evaluate(
        stage1_model_path=args.stage1_model,
        stage2_model_path=args.stage2_model,
        test_data_path=args.test_data,
        config_path=args.config,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
