"""Dataset classes for Vietnamese ABSA."""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import json


class AspectExtractionDataset(Dataset):
    """
    Dataset for aspect extraction (NER task).
    
    Converts aspect annotations to BIO tags: B-ASP, I-ASP, O
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "vinai/phobert-large",
        max_length: int = 256,
        label_map: Optional[Dict[str, int]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSON data file
            tokenizer_name: Pretrained tokenizer name
            max_length: Maximum sequence length
            label_map: Mapping from label names to IDs
        """
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Default label map
        self.label_map = label_map or {"O": 0, "B-ASP": 1, "I-ASP": 2}
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _create_ner_labels(self, text: str, aspects: List[Dict]) -> List[str]:
        """
        Create BIO labels from aspect annotations.
        
        Args:
            text: Input text
            aspects: List of aspect dictionaries with 'term' and 'span'
            
        Returns:
            List of BIO labels for each character
        """
        # Initialize all as 'O'
        char_labels = ['O'] * len(text)
        
        # Mark aspect spans
        for aspect in aspects:
            span = aspect['span']
            start, end = span[0], span[1]
            
            if start < len(text) and end <= len(text):
                # Mark first character as B-ASP
                char_labels[start] = 'B-ASP'
                # Mark remaining characters as I-ASP
                for i in range(start + 1, end):
                    char_labels[i] = 'I-ASP'
        
        return char_labels
    
    def _align_labels_with_tokens(
        self,
        text: str,
        char_labels: List[str],
        encoding
    ) -> List[int]:
        """
        Align character-level labels with subword tokens.
        
        Args:
            text: Original text
            char_labels: Character-level BIO labels
            encoding: Tokenizer encoding
            
        Returns:
            Token-level label IDs
        """
        token_labels = []
        
        for token_idx in range(len(encoding.input_ids)):
            # Get character span for this token
            span = encoding.token_to_chars(token_idx)
            
            if span is None:
                # Special tokens get 'O' label
                token_labels.append(self.label_map['O'])
            else:
                # Get label from first character of token
                char_idx = span.start
                if char_idx < len(char_labels):
                    label = char_labels[char_idx]
                    token_labels.append(self.label_map[label])
                else:
                    token_labels.append(self.label_map['O'])
        
        return token_labels
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        item = self.data[idx]
        text = item.get('processed_text', item['text'])
        aspects = item.get('aspects', [])
        
        # Create character-level labels
        char_labels = self._create_ner_labels(text, aspects)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Align labels with tokens
        token_labels = self._align_labels_with_tokens(text, char_labels, encoding)
        
        # Pad labels to max_length
        token_labels = token_labels + [self.label_map['O']] * (self.max_length - len(token_labels))
        token_labels = token_labels[:self.max_length]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(token_labels, dtype=torch.long)
        }


class AspectSentimentDataset(Dataset):
    """
    Dataset for aspect sentiment classification.
    
    Input format: [CLS] text [SEP] aspect [SEP]
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "vinai/phobert-large",
        max_length: int = 256,
        label_map: Optional[Dict[str, int]] = None,
        use_negation_mask: bool = False,
        use_intensity_mask: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSON data file
            tokenizer_name: Pretrained tokenizer name
            max_length: Maximum sequence length
            label_map: Mapping from sentiment labels to IDs
            use_negation_mask: Whether to include negation mask
            use_intensity_mask: Whether to include intensity mask
        """
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.use_negation_mask = use_negation_mask
        self.use_intensity_mask = use_intensity_mask
        
        # Default label map
        self.label_map = label_map or {
            "positive": 0, "negative": 1, "neutral": 2, "mixed": 3
        }
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        
        # Load and flatten data
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.data = self._flatten_aspects(raw_data)
    
    def _flatten_aspects(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Flatten aspect-level data into individual samples.
        
        Args:
            raw_data: List of documents with multiple aspects
            
        Returns:
            List of (text, aspect, sentiment) samples
        """
        flattened = []
        
        for item in raw_data:
            text = item.get('processed_text', item['text'])
            aspects = item.get('aspects', [])
            
            for aspect in aspects:
                aspect_term = aspect['term']
                sentiment = aspect['sentiment']
                
                flattened.append({
                    'text': text,
                    'aspect': aspect_term,
                    'sentiment': sentiment,
                    'negation_mask': item.get('negation_mask', None),
                    'intensity_mask': item.get('intensity_mask', None)
                })
        
        return flattened
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with input_ids, attention_mask, labels, and optional masks
        """
        item = self.data[idx]
        text = item['text']
        aspect = item['aspect']
        sentiment = item['sentiment']
        
        # Create input: [CLS] text [SEP] aspect [SEP]
        encoding = self.tokenizer(
            text,
            aspect,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.label_map[sentiment], dtype=torch.long)
        }
        
        # Add negation mask if available
        if self.use_negation_mask and item['negation_mask'] is not None:
            negation_mask = item['negation_mask']
            # Pad/truncate to max_length
            negation_mask = negation_mask + [0] * (self.max_length - len(negation_mask))
            negation_mask = negation_mask[:self.max_length]
            result['negation_mask'] = torch.tensor(negation_mask, dtype=torch.float)
        
        # Add intensity mask if available
        if self.use_intensity_mask and item['intensity_mask'] is not None:
            intensity_mask = item['intensity_mask']
            # Pad/truncate to max_length
            intensity_mask = intensity_mask + [0] * (self.max_length - len(intensity_mask))
            intensity_mask = intensity_mask[:self.max_length]
            result['intensity_mask'] = torch.tensor(intensity_mask, dtype=torch.float)
        
        return result


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 16,
    num_workers: int = 4
) -> Tuple:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of data loaders
    """
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader
