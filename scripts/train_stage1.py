#!/usr/bin/env python3
"""Train Stage 1: Aspect Extraction Model."""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, TrainingArguments
from src.models import create_aspect_extraction_model
from src.dataset import AspectExtractionDataset
from src.training import create_trainer, compute_ner_metrics, save_model_and_tokenizer
from src.utils import load_config, set_seed, print_training_info, get_device


def train_stage1(config_path: str = "config/training_stage1.yaml"):
    """
    Train Stage 1 aspect extraction model.
    
    Args:
        config_path: Path to training configuration
    """
    print("="*80)
    print("STAGE 1: ASPECT EXTRACTION TRAINING")
    print("="*80)
    
    # Load config
    config = load_config(config_path)
    
    # Set seed
    seed = config['training'].get('seed', 42)
    set_seed(seed)
    
    # Get device
    device = get_device()
    
    # Load tokenizer
    model_name = config['model']['backbone']
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = AspectExtractionDataset(
        data_path=config['data']['train_file'],
        tokenizer_name=model_name,
        max_length=config['model']['max_length'],
        label_map=config.get('label_map')
    )
    
    val_dataset = AspectExtractionDataset(
        data_path=config['data']['val_file'],
        tokenizer_name=model_name,
        max_length=config['model']['max_length'],
        label_map=config.get('label_map')
    )
    
    # Create model
    print(f"\nCreating model: {model_name}")
    model = create_aspect_extraction_model(
        model_name=model_name,
        num_labels=config['model']['num_labels']
    )
    
    # Print training info
    print_training_info(config, model, train_dataset, val_dataset)
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_ratio=config['training']['warmup_ratio'],
        fp16=config['training']['fp16'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        logging_steps=config['training']['logging_steps'],
        logging_dir=config['training']['logging_dir'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training']['greater_is_better'],
        seed=seed,
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        remove_unused_columns=config['training']['remove_unused_columns'],
        report_to="none"  # Disable wandb, tensorboard, etc.
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = create_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_ner_metrics,
        early_stopping_patience=config['training'].get('early_stopping_patience', 3)
    )
    
    # Train
    print("\nStarting training...")
    train_result = trainer.train()
    
    # Save final model
    print("\nSaving model...")
    save_model_and_tokenizer(model, tokenizer, config['training']['output_dir'])
    
    # Evaluate
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best model saved to: {config['training']['output_dir']}")
    print(f"Validation F1: {eval_results.get('eval_f1', 0):.4f}")
    print(f"Validation Precision: {eval_results.get('eval_precision', 0):.4f}")
    print(f"Validation Recall: {eval_results.get('eval_recall', 0):.4f}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train Stage 1: Aspect Extraction')
    parser.add_argument('--config', type=str, default='config/training_stage1.yaml',
                       help='Path to training configuration')
    
    args = parser.parse_args()
    
    train_stage1(args.config)


if __name__ == '__main__':
    main()
