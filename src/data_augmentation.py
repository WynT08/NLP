"""Data augmentation for Vietnamese text."""

import random
import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np


class VietnameseAugmenter:
    """Data augmentation for Vietnamese text."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize augmenter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        aug_config = self.config.get('augmentation', {})
        
        # Augmentation settings
        self.bt_enabled = aug_config.get('back_translation', {}).get('enabled', True)
        self.bt_prob = aug_config.get('back_translation', {}).get('probability', 0.3)
        
        self.mlm_enabled = aug_config.get('contextual_replacement', {}).get('enabled', True)
        self.mlm_prob = aug_config.get('contextual_replacement', {}).get('probability', 0.3)
        self.mask_prob = aug_config.get('contextual_replacement', {}).get('mask_probability', 0.15)
        
        self.random_enabled = aug_config.get('random_operations', {}).get('enabled', True)
        self.swap_prob = aug_config.get('random_operations', {}).get('swap_probability', 0.1)
        self.delete_prob = aug_config.get('random_operations', {}).get('delete_probability', 0.1)
        
        # Lazy load models
        self._mlm_model = None
        self._mlm_tokenizer = None
    
    def _get_mlm_model(self):
        """Lazy load PhoBERT MLM model."""
        if self._mlm_model is None:
            print("Loading PhoBERT for contextual replacement...")
            self._mlm_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            self._mlm_model = AutoModelForMaskedLM.from_pretrained("vinai/phobert-base")
            self._mlm_model.eval()
            
            if torch.cuda.is_available():
                self._mlm_model = self._mlm_model.cuda()
        
        return self._mlm_model, self._mlm_tokenizer
    
    def back_translate(self, text: str, intermediate_lang: str = 'en') -> str:
        """
        Back-translation augmentation (placeholder).
        
        Note: This requires translation API (Google Translate, etc.)
        For production, integrate with translation service.
        
        Args:
            text: Input text
            intermediate_lang: Intermediate language for translation
            
        Returns:
            Back-translated text
        """
        # Placeholder - in production, use translation API
        # Example: Vietnamese -> English -> Vietnamese
        return text
    
    def contextual_word_replacement(self, text: str, aspect_spans: List[Tuple] = None) -> str:
        """
        Replace words using PhoBERT MLM predictions.
        
        Args:
            text: Input text
            aspect_spans: List of (start, end) spans to protect
            
        Returns:
            Augmented text
        """
        if not self.mlm_enabled:
            return text
        
        model, tokenizer = self._get_mlm_model()
        words = text.split()
        
        if len(words) < 2:
            return text
        
        # Randomly select words to mask (avoid aspect terms)
        protected_indices = set()
        if aspect_spans:
            # Mark protected word indices
            for start, end in aspect_spans:
                protected_indices.update(range(start, end))
        
        # Select random word to replace
        candidates = [i for i in range(len(words)) if i not in protected_indices]
        if not candidates:
            return text
        
        mask_idx = random.choice(candidates)
        masked_words = words.copy()
        masked_words[mask_idx] = tokenizer.mask_token
        masked_text = ' '.join(masked_words)
        
        # Predict masked word
        inputs = tokenizer(masked_text, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits
        
        # Get top-k predictions for masked position
        mask_token_index = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        if len(mask_token_index) > 0:
            mask_token_logits = predictions[0, mask_token_index[0], :]
            top_tokens = torch.topk(mask_token_logits, k=5).indices.tolist()
            
            # Select random token from top-k
            replacement_token_id = random.choice(top_tokens)
            replacement_word = tokenizer.decode([replacement_token_id]).strip()
            
            words[mask_idx] = replacement_word
        
        return ' '.join(words)
    
    def random_swap(self, text: str) -> str:
        """
        Randomly swap two words.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) < 2:
            return text
        
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_deletion(self, text: str) -> str:
        """
        Randomly delete a word.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) < 2:
            return text
        
        idx = random.randint(0, len(words) - 1)
        words.pop(idx)
        
        return ' '.join(words)
    
    def augment(self, text: str, aspects: List[Dict] = None, num_augmentations: int = 1) -> List[str]:
        """
        Apply multiple augmentation techniques.
        
        Args:
            text: Input text
            aspects: List of aspect dictionaries (to protect spans)
            num_augmentations: Number of augmented versions to generate
            
        Returns:
            List of augmented texts
        """
        augmented_texts = []
        
        # Extract aspect spans to protect
        aspect_spans = []
        if aspects:
            for aspect in aspects:
                if 'span' in aspect:
                    aspect_spans.append(tuple(aspect['span']))
        
        for _ in range(num_augmentations):
            aug_text = text
            
            # Apply random augmentations
            if self.mlm_enabled and random.random() < self.mlm_prob:
                aug_text = self.contextual_word_replacement(aug_text, aspect_spans)
            
            if self.bt_enabled and random.random() < self.bt_prob:
                aug_text = self.back_translate(aug_text)
            
            if self.random_enabled:
                if random.random() < self.swap_prob:
                    aug_text = self.random_swap(aug_text)
                
                if random.random() < self.delete_prob and len(aug_text.split()) > 2:
                    aug_text = self.random_deletion(aug_text)
            
            if aug_text != text:
                augmented_texts.append(aug_text)
        
        return augmented_texts
    
    def augment_dataset(
        self,
        dataset: List[Dict],
        target_size: Optional[int] = None,
        augmentation_factor: float = 1.3
    ) -> List[Dict]:
        """
        Augment entire dataset.
        
        Args:
            dataset: List of data samples
            target_size: Target dataset size (if None, use augmentation_factor)
            augmentation_factor: Multiplier for dataset size
            
        Returns:
            Augmented dataset
        """
        if target_size is None:
            target_size = int(len(dataset) * augmentation_factor)
        
        augmented_dataset = dataset.copy()
        num_to_generate = target_size - len(dataset)
        
        print(f"Generating {num_to_generate} augmented samples...")
        
        generated = 0
        while generated < num_to_generate:
            # Randomly select sample to augment
            sample = random.choice(dataset)
            text = sample['text']
            aspects = sample.get('aspects', [])
            
            # Generate augmented version
            aug_texts = self.augment(text, aspects, num_augmentations=1)
            
            if aug_texts:
                aug_sample = sample.copy()
                aug_sample['text'] = aug_texts[0]
                augmented_dataset.append(aug_sample)
                generated += 1
            
            if (generated + 1) % 100 == 0:
                print(f"Generated {generated}/{num_to_generate} samples")
        
        print(f"Dataset augmented from {len(dataset)} to {len(augmented_dataset)} samples")
        return augmented_dataset


def create_augmenter(config_path: str = "config/config.yaml") -> VietnameseAugmenter:
    """
    Create augmenter from config file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized augmenter
    """
    from .utils import load_config
    
    config = load_config(config_path)
    return VietnameseAugmenter(config)
