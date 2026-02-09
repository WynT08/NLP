"""Tests for model architectures."""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import AspectExtractionModel, AspectSentimentModel
from transformers import AutoConfig


def test_aspect_extraction_model_forward():
    """Test aspect extraction model forward pass."""
    # Create small config for testing
    config = AutoConfig.from_pretrained("vinai/phobert-base")
    config.hidden_size = 128
    config.num_hidden_layers = 2
    
    model = AspectExtractionModel(config, num_labels=3)
    
    # Create dummy input
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, 3, (batch_size, seq_length))
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    assert 'loss' in outputs
    assert 'logits' in outputs
    assert 'predictions' in outputs
    assert outputs['logits'].shape == (batch_size, seq_length, 3)
    assert outputs['loss'] is not None


def test_aspect_sentiment_model_forward():
    """Test aspect sentiment model forward pass."""
    # Create small config for testing
    config = AutoConfig.from_pretrained("vinai/phobert-base")
    config.hidden_size = 128
    config.num_hidden_layers = 2
    
    model = AspectSentimentModel(config, num_labels=4)
    
    # Create dummy input
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, 4, (batch_size,))
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    assert 'loss' in outputs
    assert 'logits' in outputs
    assert 'predictions' in outputs
    assert outputs['logits'].shape == (batch_size, 4)
    assert outputs['loss'] is not None


def test_aspect_sentiment_model_with_features():
    """Test aspect sentiment model with negation/intensity features."""
    config = AutoConfig.from_pretrained("vinai/phobert-base")
    config.hidden_size = 128
    config.num_hidden_layers = 2
    
    model = AspectSentimentModel(
        config,
        num_labels=4,
        use_negation_features=True,
        use_intensity_features=True
    )
    
    # Create dummy input
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    negation_mask = torch.randint(0, 2, (batch_size, seq_length)).float()
    intensity_mask = torch.randint(0, 2, (batch_size, seq_length)).float()
    labels = torch.randint(0, 4, (batch_size,))
    
    # Forward pass with features
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        negation_mask=negation_mask,
        intensity_mask=intensity_mask,
        labels=labels
    )
    
    assert 'loss' in outputs
    assert 'logits' in outputs
    assert outputs['logits'].shape == (batch_size, 4)


def test_model_num_labels():
    """Test model with different number of labels."""
    config = AutoConfig.from_pretrained("vinai/phobert-base")
    config.hidden_size = 128
    config.num_hidden_layers = 2
    
    # Test aspect extraction with 3 labels
    model1 = AspectExtractionModel(config, num_labels=3)
    assert model1.num_labels == 3
    
    # Test sentiment with 4 labels
    model2 = AspectSentimentModel(config, num_labels=4)
    assert model2.num_labels == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
