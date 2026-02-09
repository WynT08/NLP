"""Evaluation metrics for Vietnamese ABSA."""

import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)


def evaluate_ner(true_labels: List[List[str]], pred_labels: List[List[str]]) -> Dict[str, float]:
    """
    Evaluate NER predictions using seqeval.
    
    Args:
        true_labels: True BIO labels
        pred_labels: Predicted BIO labels
        
    Returns:
        Dictionary of metrics
    """
    from seqeval.metrics import (
        f1_score, precision_score, recall_score,
        classification_report as seqeval_report
    )
    
    metrics = {
        'precision': precision_score(true_labels, pred_labels),
        'recall': recall_score(true_labels, pred_labels),
        'f1': f1_score(true_labels, pred_labels)
    }
    
    # Detailed report
    report = seqeval_report(true_labels, pred_labels, output_dict=True)
    
    return metrics, report


def evaluate_classification(
    true_labels: List[int],
    pred_labels: List[int],
    label_names: List[str]
) -> Dict[str, float]:
    """
    Evaluate classification predictions.
    
    Args:
        true_labels: True labels
        pred_labels: Predicted labels
        label_names: Label names
        
    Returns:
        Dictionary of metrics and report
    """
    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'macro_f1': f1_score(true_labels, pred_labels, average='macro'),
        'weighted_f1': f1_score(true_labels, pred_labels, average='weighted'),
        'macro_precision': precision_score(true_labels, pred_labels, average='macro'),
        'macro_recall': recall_score(true_labels, pred_labels, average='macro')
    }
    
    # Per-class metrics
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=label_names,
        output_dict=True
    )
    
    return metrics, report


def plot_confusion_matrix(
    true_labels: List[int],
    pred_labels: List[int],
    label_names: List[str],
    save_path: str = None
):
    """
    Plot confusion matrix.
    
    Args:
        true_labels: True labels
        pred_labels: Predicted labels
        label_names: Label names
        save_path: Path to save plot
    """
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_absa_end_to_end(
    predictions: List[Dict],
    ground_truth: List[Dict]
) -> Dict[str, float]:
    """
    Evaluate end-to-end ABSA performance.
    
    Args:
        predictions: List of predicted results
        ground_truth: List of ground truth annotations
        
    Returns:
        Dictionary of metrics
    """
    aspect_matches = 0
    exact_matches = 0
    relaxed_matches = 0
    total_pred_aspects = 0
    total_true_aspects = 0
    
    for pred, true in zip(predictions, ground_truth):
        pred_aspects = {a['aspect']: a['sentiment'] for a in pred.get('aspects', [])}
        true_aspects = {a['term']: a['sentiment'] for a in true.get('aspects', [])}
        
        total_pred_aspects += len(pred_aspects)
        total_true_aspects += len(true_aspects)
        
        # Count matches
        for aspect, sentiment in pred_aspects.items():
            # Exact match: aspect term and sentiment both correct
            if aspect in true_aspects and true_aspects[aspect] == sentiment:
                exact_matches += 1
                aspect_matches += 1
            # Aspect match: only aspect term is correct
            elif aspect in true_aspects:
                aspect_matches += 1
            # Relaxed match: partial overlap in aspect term
            else:
                for true_aspect in true_aspects.keys():
                    if aspect in true_aspect or true_aspect in aspect:
                        relaxed_matches += 1
                        break
    
    # Calculate metrics
    aspect_precision = aspect_matches / total_pred_aspects if total_pred_aspects > 0 else 0
    aspect_recall = aspect_matches / total_true_aspects if total_true_aspects > 0 else 0
    aspect_f1 = 2 * aspect_precision * aspect_recall / (aspect_precision + aspect_recall) if (aspect_precision + aspect_recall) > 0 else 0
    
    exact_match_rate = exact_matches / total_true_aspects if total_true_aspects > 0 else 0
    relaxed_match_rate = (aspect_matches + relaxed_matches) / total_true_aspects if total_true_aspects > 0 else 0
    
    return {
        'aspect_precision': aspect_precision,
        'aspect_recall': aspect_recall,
        'aspect_f1': aspect_f1,
        'exact_match_rate': exact_match_rate,
        'relaxed_match_rate': relaxed_match_rate,
        'total_pred_aspects': total_pred_aspects,
        'total_true_aspects': total_true_aspects,
        'exact_matches': exact_matches
    }


def print_evaluation_results(metrics: Dict[str, float], report: Dict = None):
    """
    Print evaluation results in a formatted manner.
    
    Args:
        metrics: Dictionary of metrics
        report: Optional detailed report
    """
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name}: {value:.4f}")
        else:
            print(f"{metric_name}: {value}")
    
    if report:
        print("\n" + "-"*80)
        print("DETAILED REPORT")
        print("-"*80)
        
        for label, scores in report.items():
            if isinstance(scores, dict):
                print(f"\n{label}:")
                for score_name, score_value in scores.items():
                    if isinstance(score_value, float):
                        print(f"  {score_name}: {score_value:.4f}")
                    else:
                        print(f"  {score_name}: {score_value}")
    
    print("="*80 + "\n")


class ABSAEvaluator:
    """Evaluator for ABSA pipeline."""
    
    def __init__(self, pipeline):
        """
        Initialize evaluator.
        
        Args:
            pipeline: ABSA pipeline to evaluate
        """
        self.pipeline = pipeline
    
    def evaluate_dataset(self, test_data: List[Dict]) -> Dict[str, float]:
        """
        Evaluate pipeline on test dataset.
        
        Args:
            test_data: List of test samples
            
        Returns:
            Dictionary of metrics
        """
        predictions = []
        ground_truth = []
        
        print(f"Evaluating on {len(test_data)} samples...")
        
        for i, item in enumerate(test_data):
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(test_data)} samples")
            
            text = item['text']
            pred = self.pipeline.predict(text)
            
            predictions.append(pred)
            ground_truth.append(item)
        
        # Evaluate end-to-end
        absa_metrics = evaluate_absa_end_to_end(predictions, ground_truth)
        
        print_evaluation_results(absa_metrics)
        
        return absa_metrics, predictions


def save_evaluation_results(metrics: Dict, predictions: List[Dict], output_path: str):
    """
    Save evaluation results to file.
    
    Args:
        metrics: Evaluation metrics
        predictions: Predictions
        output_path: Output file path
    """
    import json
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'metrics': metrics,
        'predictions': predictions
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation results saved to {output_path}")
