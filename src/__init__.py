"""Vietnamese ABSA package initialization."""

__version__ = "1.0.0"
__author__ = "Vietnamese ABSA Team"

from .utils import set_seed, load_config, get_device
from .preprocessing import VietnamesePreprocessor, create_preprocessor
from .models import (
    AspectExtractionModel,
    AspectSentimentModel,
    create_aspect_extraction_model,
    create_aspect_sentiment_model
)
from .dataset import AspectExtractionDataset, AspectSentimentDataset
from .training import (
    WeightedTrainer,
    compute_ner_metrics,
    compute_classification_metrics,
    create_trainer
)
from .inference import ABSAPipeline, create_pipeline
from .evaluation import (
    evaluate_ner,
    evaluate_classification,
    evaluate_absa_end_to_end,
    ABSAEvaluator
)
from .data_augmentation import VietnameseAugmenter, create_augmenter
from .lexicon_validator import LexiconValidator, validate_lexicon

__all__ = [
    # Utils
    'set_seed',
    'load_config',
    'get_device',
    
    # Preprocessing
    'VietnamesePreprocessor',
    'create_preprocessor',
    
    # Models
    'AspectExtractionModel',
    'AspectSentimentModel',
    'create_aspect_extraction_model',
    'create_aspect_sentiment_model',
    
    # Dataset
    'AspectExtractionDataset',
    'AspectSentimentDataset',
    
    # Training
    'WeightedTrainer',
    'compute_ner_metrics',
    'compute_classification_metrics',
    'create_trainer',
    
    # Inference
    'ABSAPipeline',
    'create_pipeline',
    
    # Evaluation
    'evaluate_ner',
    'evaluate_classification',
    'evaluate_absa_end_to_end',
    'ABSAEvaluator',
    
    # Data augmentation
    'VietnameseAugmenter',
    'create_augmenter',
    
    # Lexicon validator
    'LexiconValidator',
    'validate_lexicon',
]
