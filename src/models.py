"""Model architectures for Vietnamese ABSA."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from typing import Optional, Tuple, Dict


class AspectExtractionModel(PreTrainedModel):
    """
    PhoBERT-based model for aspect extraction (NER).
    
    Predicts B-ASP, I-ASP, O tags for aspect term extraction.
    """
    
    def __init__(self, config, num_labels: int = 3):
        """
        Initialize aspect extraction model.
        
        Args:
            config: Model configuration
            num_labels: Number of NER labels (default: 3 for B-ASP, I-ASP, O)
        """
        super().__init__(config)
        self.num_labels = num_labels
        
        # PhoBERT backbone
        self.roberta = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: NER labels [batch_size, seq_len]
            
        Returns:
            Dictionary with loss, logits, and predictions
        """
        # Get embeddings from PhoBERT
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)
        
        # Classification
        logits = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            
            # Only compute loss on real tokens (not padding)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'predictions': torch.argmax(logits, dim=-1)
        }


class AspectSentimentModel(PreTrainedModel):
    """
    PhoBERT-based model for aspect sentiment classification.
    
    Input format: [CLS] text [SEP] aspect [SEP]
    Predicts: positive, negative, neutral, mixed
    """
    
    def __init__(
        self, 
        config, 
        num_labels: int = 4,
        use_negation_features: bool = False,
        use_intensity_features: bool = False
    ):
        """
        Initialize aspect sentiment model.
        
        Args:
            config: Model configuration
            num_labels: Number of sentiment labels (default: 4)
            use_negation_features: Whether to use negation mask features
            use_intensity_features: Whether to use intensity mask features
        """
        super().__init__(config)
        self.num_labels = num_labels
        self.use_negation_features = use_negation_features
        self.use_intensity_features = use_intensity_features
        
        # PhoBERT backbone
        self.roberta = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Feature size
        feature_size = config.hidden_size
        if use_negation_features:
            feature_size += config.hidden_size
        if use_intensity_features:
            feature_size += config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, num_labels)
        )
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negation_mask: Optional[torch.Tensor] = None,
        intensity_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            negation_mask: Binary negation mask [batch_size, seq_len]
            intensity_mask: Binary intensity mask [batch_size, seq_len]
            labels: Sentiment labels [batch_size]
            
        Returns:
            Dictionary with loss, logits, and predictions
        """
        # Get embeddings from PhoBERT
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get [CLS] representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        
        # Optionally add feature representations
        features = [pooled_output]
        
        if self.use_negation_features and negation_mask is not None:
            # Average pooling over negation-marked tokens
            negation_mask_expanded = negation_mask.unsqueeze(-1).float()
            negation_features = (outputs.last_hidden_state * negation_mask_expanded).sum(1)
            negation_count = negation_mask.sum(1, keepdim=True).clamp(min=1)
            negation_features = negation_features / negation_count
            features.append(negation_features)
        
        if self.use_intensity_features and intensity_mask is not None:
            # Average pooling over intensity-marked tokens
            intensity_mask_expanded = intensity_mask.unsqueeze(-1).float()
            intensity_features = (outputs.last_hidden_state * intensity_mask_expanded).sum(1)
            intensity_count = intensity_mask.sum(1, keepdim=True).clamp(min=1)
            intensity_features = intensity_features / intensity_count
            features.append(intensity_features)
        
        # Concatenate features
        combined_features = torch.cat(features, dim=-1)
        
        # Classification
        logits = self.classifier(combined_features)  # [batch_size, num_labels]
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'predictions': torch.argmax(logits, dim=-1)
        }


def create_aspect_extraction_model(
    model_name: str = "vinai/phobert-large",
    num_labels: int = 3
) -> AspectExtractionModel:
    """
    Create aspect extraction model.
    
    Args:
        model_name: Pretrained model name
        num_labels: Number of NER labels
        
    Returns:
        Initialized model
    """
    config = AutoConfig.from_pretrained(model_name)
    model = AspectExtractionModel(config, num_labels=num_labels)
    
    # Load pretrained weights for backbone
    pretrained = AutoModel.from_pretrained(model_name)
    model.roberta = pretrained
    
    return model


def create_aspect_sentiment_model(
    model_name: str = "vinai/phobert-large",
    num_labels: int = 4,
    use_negation_features: bool = False,
    use_intensity_features: bool = False
) -> AspectSentimentModel:
    """
    Create aspect sentiment model.
    
    Args:
        model_name: Pretrained model name
        num_labels: Number of sentiment labels
        use_negation_features: Whether to use negation features
        use_intensity_features: Whether to use intensity features
        
    Returns:
        Initialized model
    """
    config = AutoConfig.from_pretrained(model_name)
    model = AspectSentimentModel(
        config,
        num_labels=num_labels,
        use_negation_features=use_negation_features,
        use_intensity_features=use_intensity_features
    )
    
    # Load pretrained weights for backbone
    pretrained = AutoModel.from_pretrained(model_name)
    model.roberta = pretrained
    
    return model
