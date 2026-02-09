#!/usr/bin/env python3
"""Train Stage 2: Aspect Sentiment Classification Model."""

import argparse
from pathlib import Path
import sys
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, TrainingArguments
from src.models import create_aspect_sentiment_model
from src.dataset import AspectSentimentDataset
from src.training import create_trainer, compute_classification_metrics, save_model_and_tokenizer
from src.utils import load_config, set_seed, print_training_info, get_device, calculate_class_weights, load_json


def train_stage2(config_path: str = "config/training_stage2.yaml"):
    """
    Train Stage 2 aspect sentiment classification model.
    
    Args:
        config_path: Path to training configuration
    """
    print("="*80)
    print("STAGE 2: ASPECT SENTIMENT CLASSIFICATION TRAINING")
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
    use_negation = config['model'].get('use_negation_features', False)
    use_intensity = config['model'].get('use_intensity_features', False)
    
    train_dataset = AspectSentimentDataset(
        data_path=config['data']['train_file'],
        tokenizer_name=model_name,
        max_length=config['model']['max_length'],
        label_map=config.get('label_map'),
        use_negation_mask=use_negation,
        use_intensity_mask=use_intensity
    )
    
    val_dataset = AspectSentimentDataset(
        data_path=config['data']['val_file'],
        tokenizer_name=model_name,
        max_length=config['model']['max_length'],
        label_map=config.get('label_map'),
        use_negation_mask=use_negation,
        use_intensity_mask=use_intensity
    )
    
    # Calculate class weights if enabled
    class_weights = None
    if config['training'].get('use_class_weights', False):
        print("\nCalculating class weights...")
        
        if 'class_weights' in config['training']:
            # Use predefined weights
            class_weights = torch.tensor(config['training']['class_weights'], dtype=torch.float32)
            print(f"Using predefined class weights: {class_weights.tolist()}")
        else:
            # Calculate from training data
            train_labels = [item['labels'].item() for item in train_dataset]
            class_weights = calculate_class_weights(train_labels, config['model']['num_labels'])
            print(f"Calculated class weights: {class_weights.tolist()}")
    
    # Create model
    print(f"\nCreating model: {model_name}")
    model = create_aspect_sentiment_model(
        model_name=model_name,
        num_labels=config['model']['num_labels'],
        use_negation_features=use_negation,
        use_intensity_features=use_intensity
    )
    
    # Print training info
    print_training_info(config, model, train_dataset, val_dataset)
    
    if use_negation:
        print("Using negation features")
    if use_intensity:
        print("Using intensity features")
    
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
        compute_metrics=compute_classification_metrics,
        class_weights=class_weights,
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
    print(f"Validation Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
    print(f"Validation Macro-F1: {eval_results.get('eval_macro_f1', 0):.4f}")
    print(f"Validation Weighted-F1: {eval_results.get('eval_weighted_f1', 0):.4f}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train Stage 2: Aspect Sentiment Classification')
    parser.add_argument('--config', type=str, default='config/training_stage2.yaml',
                       help='Path to training configuration')
    
    args = parser.parse_args()
    
    train_stage2(args.config)


if __name__ == '__main__':
    main()
