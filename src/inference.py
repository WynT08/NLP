"""Inference pipeline for Vietnamese ABSA."""

import torch
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import numpy as np


class ABSAPipeline:
    """
    End-to-end ABSA inference pipeline.
    
    Combines aspect extraction and sentiment classification.
    """
    
    def __init__(
        self,
        stage1_model_path: str,
        stage2_model_path: str,
        config: Optional[Dict] = None,
        device: Optional[str] = None
    ):
        """
        Initialize ABSA pipeline.
        
        Args:
            stage1_model_path: Path to aspect extraction model
            stage2_model_path: Path to sentiment classification model
            config: Configuration dictionary
            device: Device to use (cuda/cpu)
        """
        self.config = config or {}
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load models
        print("Loading Stage 1 model (aspect extraction)...")
        from .models import AspectExtractionModel
        self.stage1_model = AspectExtractionModel.from_pretrained(stage1_model_path)
        self.stage1_model.to(self.device)
        self.stage1_model.eval()
        
        print("Loading Stage 2 model (sentiment classification)...")
        from .models import AspectSentimentModel
        use_negation = self.config.get('model', {}).get('use_negation_features', False)
        use_intensity = self.config.get('model', {}).get('use_intensity_features', False)
        
        self.stage2_model = AspectSentimentModel.from_pretrained(
            stage2_model_path,
            use_negation_features=use_negation,
            use_intensity_features=use_intensity
        )
        self.stage2_model.to(self.device)
        self.stage2_model.eval()
        
        # Load tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(stage1_model_path)
        
        # Load preprocessor
        from .preprocessing import VietnamesePreprocessor
        self.preprocessor = VietnamesePreprocessor(self.config)
        
        # Label maps
        self.ner_labels = {0: "O", 1: "B-ASP", 2: "I-ASP"}
        self.sentiment_labels = {0: "positive", 1: "negative", 2: "neutral", 3: "mixed"}
        
        print("Pipeline initialized successfully!")
    
    def extract_aspects(self, text: str) -> List[str]:
        """
        Extract aspect terms from text using Stage 1 model.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted aspect terms
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = self.stage1_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            else:
                outputs = self.stage1_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
        
        predictions = outputs['predictions'].cpu().numpy()[0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        
        # Extract aspects from BIO tags
        aspects = []
        current_aspect = []
        
        for token, pred in zip(tokens, predictions):
            if token in ['<s>', '</s>', '<pad>']:
                continue
            
            label = self.ner_labels[pred]
            
            if label == 'B-ASP':
                if current_aspect:
                    aspect_text = self._merge_tokens(current_aspect)
                    if aspect_text:
                        aspects.append(aspect_text)
                current_aspect = [token]
            elif label == 'I-ASP':
                if current_aspect:
                    current_aspect.append(token)
            else:  # O
                if current_aspect:
                    aspect_text = self._merge_tokens(current_aspect)
                    if aspect_text:
                        aspects.append(aspect_text)
                    current_aspect = []
        
        # Handle last aspect
        if current_aspect:
            aspect_text = self._merge_tokens(current_aspect)
            if aspect_text:
                aspects.append(aspect_text)
        
        return aspects
    
    def _merge_tokens(self, tokens: List[str]) -> str:
        """Merge subword tokens into text."""
        text = ""
        for token in tokens:
            if token.startswith("##"):
                text += token[2:]
            elif token.startswith("▁"):
                text += " " + token[1:]
            else:
                text += token
        return text.strip()
    
    def classify_aspect_sentiment(
        self,
        text: str,
        aspect: str,
        negation_mask: Optional[List[int]] = None,
        intensity_mask: Optional[List[int]] = None
    ) -> Tuple[str, float]:
        """
        Classify sentiment for a given aspect.
        
        Args:
            text: Input text
            aspect: Aspect term
            negation_mask: Optional negation mask
            intensity_mask: Optional intensity mask
            
        Returns:
            Tuple of (sentiment, confidence)
        """
        # Tokenize text and aspect
        encoding = self.tokenizer(
            text,
            aspect,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Prepare optional masks
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        if negation_mask is not None:
            negation_tensor = torch.tensor([negation_mask], dtype=torch.float).to(self.device)
            batch['negation_mask'] = negation_tensor
        
        if intensity_mask is not None:
            intensity_tensor = torch.tensor([intensity_mask], dtype=torch.float).to(self.device)
            batch['intensity_mask'] = intensity_tensor
        
        # Predict
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = self.stage2_model(**batch)
            else:
                outputs = self.stage2_model(**batch)
        
        logits = outputs['logits'].cpu().numpy()[0]
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        
        pred_label_id = np.argmax(probs)
        sentiment = self.sentiment_labels[pred_label_id]
        confidence = float(probs[pred_label_id])
        
        return sentiment, confidence
    
    def aggregate_overall_sentiment(self, aspect_sentiments: List[str]) -> str:
        """
        Aggregate aspect-level sentiments to overall sentiment.
        
        Args:
            aspect_sentiments: List of aspect sentiments
            
        Returns:
            Overall sentiment (positive/negative/neutral/mixed)
        """
        if not aspect_sentiments:
            return "neutral"
        
        sentiment_counts = {
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'mixed': 0
        }
        
        for sent in aspect_sentiments:
            sentiment_counts[sent] += 1
        
        # If already has mixed, return mixed
        if sentiment_counts['mixed'] > 0:
            return "mixed"
        
        # If both positive and negative, return mixed
        if sentiment_counts['positive'] > 0 and sentiment_counts['negative'] > 0:
            return "mixed"
        
        # Return dominant sentiment
        if sentiment_counts['positive'] > sentiment_counts['negative']:
            return "positive"
        elif sentiment_counts['negative'] > sentiment_counts['positive']:
            return "negative"
        else:
            return "neutral"
    
    def predict(self, text: str) -> Dict:
        """
        Full ABSA prediction pipeline.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with aspects and sentiments
        """
        # Preprocess
        preprocessed = self.preprocessor.preprocess(text, return_masks=True)
        processed_text = preprocessed['processed_text']
        negation_mask = preprocessed.get('negation_mask')
        intensity_mask = preprocessed.get('intensity_mask')
        
        # Extract aspects
        aspects = self.extract_aspects(processed_text)
        
        # Classify sentiment for each aspect
        aspect_results = []
        aspect_sentiments = []
        
        for aspect in aspects:
            sentiment, confidence = self.classify_aspect_sentiment(
                processed_text,
                aspect,
                negation_mask=negation_mask,
                intensity_mask=intensity_mask
            )
            
            aspect_results.append({
                'aspect': aspect,
                'sentiment': sentiment,
                'confidence': confidence
            })
            aspect_sentiments.append(sentiment)
        
        # Aggregate overall sentiment
        overall_sentiment = self.aggregate_overall_sentiment(aspect_sentiments)
        
        return {
            'text': text,
            'processed_text': processed_text,
            'aspects': aspect_results,
            'overall_sentiment': overall_sentiment
        }


def create_pipeline(
    stage1_model_path: str,
    stage2_model_path: str,
    config_path: str = "config/config.yaml"
) -> ABSAPipeline:
    """
    Create ABSA pipeline from config.
    
    Args:
        stage1_model_path: Path to stage 1 model
        stage2_model_path: Path to stage 2 model
        config_path: Path to configuration file
        
    Returns:
        Initialized pipeline
    """
    from .utils import load_config
    
    config = load_config(config_path)
    return ABSAPipeline(stage1_model_path, stage2_model_path, config)
