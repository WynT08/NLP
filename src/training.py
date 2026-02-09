"""Training utilities for Vietnamese ABSA."""

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from typing import Dict, Optional
import numpy as np


class WeightedTrainer(Trainer):
    """Custom trainer with class weights for imbalanced datasets."""
    
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        """
        Initialize weighted trainer.
        
        Args:
            class_weights: Class weights for loss computation
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.args.device)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute weighted loss.
        
        Args:
            model: The model
            inputs: Input batch
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss tensor or tuple of (loss, outputs)
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits, labels)
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def compute_ner_metrics(p) -> Dict[str, float]:
    """
    Compute NER metrics using seqeval.
    
    Args:
        p: Predictions object with predictions and label_ids
        
    Returns:
        Dictionary of metrics
    """
    from seqeval.metrics import f1_score, precision_score, recall_score
    
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Convert to label names
    label_map = {0: "O", 1: "B-ASP", 2: "I-ASP"}
    
    true_labels = []
    pred_labels = []
    
    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_labels = []
        
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:  # Ignore padding
                true_seq.append(label_map[label])
                pred_seq_labels.append(label_map[pred])
        
        if true_seq:
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_labels)
    
    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels)
    }


def compute_classification_metrics(p) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        p: Predictions object with predictions and label_ids
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average='macro'),
        "weighted_f1": f1_score(labels, predictions, average='weighted'),
        "precision": precision_score(labels, predictions, average='macro'),
        "recall": recall_score(labels, predictions, average='macro')
    }


def create_trainer(
    model,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset,
    compute_metrics,
    class_weights: Optional[torch.Tensor] = None,
    early_stopping_patience: int = 3
):
    """
    Create trainer with callbacks.
    
    Args:
        model: Model to train
        training_args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        compute_metrics: Metrics computation function
        class_weights: Optional class weights
        early_stopping_patience: Patience for early stopping
        
    Returns:
        Configured trainer
    """
    callbacks = []
    
    # Add early stopping
    if early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
    
    # Choose trainer class
    if class_weights is not None:
        trainer_class = WeightedTrainer
        trainer_kwargs = {'class_weights': class_weights}
    else:
        trainer_class = Trainer
        trainer_kwargs = {}
    
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        **trainer_kwargs
    )
    
    return trainer


def save_model_and_tokenizer(model, tokenizer, output_dir: str):
    """
    Save model and tokenizer.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")


def load_model_and_tokenizer(model_class, model_dir: str, **model_kwargs):
    """
    Load model and tokenizer from directory.
    
    Args:
        model_class: Model class to instantiate
        model_dir: Directory with saved model
        model_kwargs: Additional model arguments
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoTokenizer
    
    model = model_class.from_pretrained(model_dir, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return model, tokenizer
