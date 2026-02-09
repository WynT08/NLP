"""Tests for preprocessing module."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import VietnamesePreprocessor


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = VietnamesePreprocessor()
    assert preprocessor is not None
    assert preprocessor.lowercase is True


def test_normalize_text():
    """Test text normalization."""
    preprocessor = VietnamesePreprocessor()
    
    # Test whitespace normalization
    text = "Điện   thoại    này"
    result = preprocessor.normalize_text(text)
    assert result == "Điện thoại này"
    
    # Test repeated character normalization
    text = "đẹpppppp"
    result = preprocessor.normalize_text(text)
    assert "p" in result
    assert result.count("p") < 5


def test_normalize_slang():
    """Test slang normalization."""
    preprocessor = VietnamesePreprocessor()
    
    text = "ko biết wá"
    result = preprocessor.normalize_slang(text)
    assert "không" in result
    assert "quá" in result


def test_detect_negation():
    """Test negation detection."""
    preprocessor = VietnamesePreprocessor()
    
    text = "không tốt lắm"
    mask = preprocessor.detect_negation(text)
    assert len(mask) == len(text.split())
    assert mask[0] == 1  # "không" should be marked


def test_detect_intensity():
    """Test intensity detection."""
    preprocessor = VietnamesePreprocessor()
    
    text = "rất tốt"
    mask = preprocessor.detect_intensity(text)
    assert len(mask) == len(text.split())
    assert mask[0] == 1  # "rất" should be marked


def test_preprocess_full():
    """Test full preprocessing pipeline."""
    preprocessor = VietnamesePreprocessor()
    
    text = "Điện thoại này KO tốt lắm!!!"
    result = preprocessor.preprocess(text, return_masks=True)
    
    assert 'original_text' in result
    assert 'processed_text' in result
    assert 'negation_mask' in result
    assert 'intensity_mask' in result
    assert result['original_text'] == text


def test_handle_emojis():
    """Test emoji handling."""
    config = {
        'preprocessing': {
            'positive_emojis': ['😊', '😃'],
            'negative_emojis': ['😢', '😞'],
            'neutral_emojis': ['😐']
        }
    }
    preprocessor = VietnamesePreprocessor(config)
    
    text = "Tốt quá 😊"
    result, emojis = preprocessor.handle_emojis(text)
    assert len(emojis) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
