"""Vietnamese text preprocessing for ABSA."""

import re
from typing import Dict, List, Optional, Set, Tuple

import emoji


class VietnamesePreprocessor:
    """Preprocessor for Vietnamese text with ABSA-specific features."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Vietnamese preprocessor.
        
        Args:
            config: Configuration dictionary with lexicons and settings
        """
        self.config = config or {}
        preprocessing_config = self.config.get('preprocessing', {})
        
        # Settings
        self.lowercase = preprocessing_config.get('lowercase', True)
        self.normalize_emojis = preprocessing_config.get('normalize_emojis', True)
        self.word_segmentation = preprocessing_config.get('word_segmentation', True)
        
        # Load lexicons
        self.positive_emojis = set(preprocessing_config.get('positive_emojis', []))
        self.negative_emojis = set(preprocessing_config.get('negative_emojis', []))
        self.neutral_emojis = set(preprocessing_config.get('neutral_emojis', []))
        
        self.negation_words = set(preprocessing_config.get('negation_words', []))
        self.intensity_words = set(preprocessing_config.get('intensity_words', []))
        self.contrast_words = set(preprocessing_config.get('contrast_words', []))
        
        # Slang dictionary
        self.slang_dict = {
            'ko': 'không', 'k': 'không', 'hog': 'không', 'hong': 'không',
            'khum': 'không', 'hem': 'không', 'hổng': 'không',
            'wá': 'quá', 'wa': 'quá', 'quá': 'quá',
            'đc': 'được', 'dc': 'được', 'dk': 'được',
            'vs': 'với', 'wá': 'quá', 'j': 'gì', 'bik': 'biết',
            'bít': 'biết', 'cx': 'cũng', 'cug': 'cũng',
            'mik': 'mình', 'mk': 'mình', 'mjk': 'mình',
            'nc': 'nước', 'nch': 'nước', 'trc': 'trước',
            'nt': 'nhắn tin', 'zai': 'giai', 'zô': 'giô',
            'vcl': 'vô cùng lớn', 'vkl': 'vô cùng lớn',
            'oki': 'ok', 'okie': 'ok', 'okii': 'ok',
            'tks': 'cảm ơn', 'thanks': 'cảm ơn', 'thank': 'cảm ơn',
            'sr': 'sorry', 'sory': 'sorry', 'hj': 'hehe',
            'uhm': 'ừm', 'uh': 'ừ', 'a': 'ah',
            'e': 'em', 'k': 'anh', 'ntn': 'như thế nào',
        }
        
        # Word segmentation (lazy load)
        self._segmenter = None
    
    def _get_segmenter(self):
        """Lazy load underthesea word segmenter."""
        if self._segmenter is None:
            try:
                from underthesea import word_tokenize
                self._segmenter = word_tokenize
            except ImportError:
                print("Warning: underthesea not installed. Word segmentation disabled.")
                self._segmenter = lambda x: x
        return self._segmenter
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Vietnamese text.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize common Vietnamese characters
        text = text.replace('òa', 'oà').replace('óa', 'oá')
        text = text.replace('ùy', 'uỳ').replace('úy', 'uý')
        
        # Normalize repeated characters (e.g., "đẹppppp" -> "đẹp")
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Normalize emoticons
        text = re.sub(r':\)', '😊', text)
        text = re.sub(r':\(', '😢', text)
        text = re.sub(r':D', '😃', text)
        text = re.sub(r';D', '😆', text)
        
        return text
    
    def normalize_slang(self, text: str) -> str:
        """
        Normalize slang/teencode to standard Vietnamese.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized slang
        """
        words = text.split()
        normalized_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.slang_dict:
                normalized_words.append(self.slang_dict[word_lower])
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def handle_emojis(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract and optionally replace emojis.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (processed text, list of extracted emojis)
        """
        extracted_emojis = []
        
        # Extract emojis
        for char in text:
            if char in emoji.EMOJI_DATA:
                extracted_emojis.append(char)
        
        if self.normalize_emojis:
            # Replace emojis with sentiment tags
            for em in extracted_emojis:
                if em in self.positive_emojis:
                    text = text.replace(em, ' [POS_EMOJI] ')
                elif em in self.negative_emojis:
                    text = text.replace(em, ' [NEG_EMOJI] ')
                elif em in self.neutral_emojis:
                    text = text.replace(em, ' [NEU_EMOJI] ')
                else:
                    text = text.replace(em, ' ')
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text, extracted_emojis
    
    def segment_words(self, text: str) -> str:
        """
        Segment Vietnamese words using underthesea.
        
        Args:
            text: Input text
            
        Returns:
            Segmented text
        """
        if not self.word_segmentation:
            return text
        
        try:
            segmenter = self._get_segmenter()
            segmented = segmenter(text)
            return ' '.join(segmented) if isinstance(segmented, list) else segmented
        except Exception as e:
            print(f"Warning: Word segmentation failed: {e}")
            return text
    
    def detect_negation(self, text: str) -> List[int]:
        """
        Detect negation words in text.
        
        Args:
            text: Input text
            
        Returns:
            Binary mask indicating negation positions
        """
        words = text.split()
        negation_mask = [0] * len(words)
        
        for i, word in enumerate(words):
            if word.lower() in self.negation_words:
                negation_mask[i] = 1
                # Mark next 3 words as being in negation scope
                for j in range(i+1, min(i+4, len(words))):
                    negation_mask[j] = 1
        
        return negation_mask
    
    def detect_intensity(self, text: str) -> List[int]:
        """
        Detect intensity words in text.
        
        Args:
            text: Input text
            
        Returns:
            Binary mask indicating intensity positions
        """
        words = text.split()
        intensity_mask = [0] * len(words)
        
        for i, word in enumerate(words):
            if word.lower() in self.intensity_words:
                intensity_mask[i] = 1
                # Mark next 2 words as being intensified
                for j in range(i+1, min(i+3, len(words))):
                    intensity_mask[j] = 1
        
        return intensity_mask
    
    def preprocess(self, text: str, return_masks: bool = False) -> Dict:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Input text
            return_masks: Whether to return negation/intensity masks
            
        Returns:
            Dictionary with processed text and optional masks
        """
        original_text = text
        
        # Normalize text
        text = self.normalize_text(text)
        
        # Handle emojis
        text, emojis = self.handle_emojis(text)
        
        # Normalize slang
        text = self.normalize_slang(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Word segmentation
        text = self.segment_words(text)
        
        result = {
            'original_text': original_text,
            'processed_text': text,
            'emojis': emojis
        }
        
        if return_masks:
            result['negation_mask'] = self.detect_negation(text)
            result['intensity_mask'] = self.detect_intensity(text)
        
        return result


def create_preprocessor(config_path: str = "config/config.yaml") -> VietnamesePreprocessor:
    """
    Create preprocessor from config file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized preprocessor
    """
    from .utils import load_config
    
    config = load_config(config_path)
    return VietnamesePreprocessor(config)
