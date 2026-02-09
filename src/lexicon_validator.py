"""Lexicon coverage validator for Vietnamese ABSA."""

from typing import Dict, List, Set, Tuple
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


class LexiconValidator:
    """Validate and analyze lexicon coverage on dataset."""
    
    def __init__(self, config: Dict):
        """
        Initialize validator.
        
        Args:
            config: Configuration with lexicons
        """
        self.config = config
        preprocessing_config = config.get('preprocessing', {})
        
        # Load lexicons
        self.negation_words = set(preprocessing_config.get('negation_words', []))
        self.intensity_words = set(preprocessing_config.get('intensity_words', []))
        self.contrast_words = set(preprocessing_config.get('contrast_words', []))
        self.positive_indicators = set(preprocessing_config.get('positive_indicators', []))
        self.negative_indicators = set(preprocessing_config.get('negative_indicators', []))
    
    def analyze_coverage(self, dataset: List[Dict]) -> Dict[str, any]:
        """
        Analyze lexicon coverage on dataset.
        
        Args:
            dataset: List of data samples
            
        Returns:
            Coverage statistics
        """
        total_samples = len(dataset)
        
        # Count samples with lexicon words
        samples_with_negation = 0
        samples_with_intensity = 0
        samples_with_contrast = 0
        samples_with_positive = 0
        samples_with_negative = 0
        
        # Track all words
        all_words = Counter()
        uncovered_sentiment_words = Counter()
        
        for sample in dataset:
            text = sample.get('processed_text', sample['text']).lower()
            words = set(text.split())
            all_words.update(words)
            
            # Check coverage
            if words & self.negation_words:
                samples_with_negation += 1
            
            if words & self.intensity_words:
                samples_with_intensity += 1
            
            if words & self.contrast_words:
                samples_with_contrast += 1
            
            if words & self.positive_indicators:
                samples_with_positive += 1
            
            if words & self.negative_indicators:
                samples_with_negative += 1
            
            # Find potentially missing sentiment words
            sentiment = sample.get('overall_sentiment', '')
            if sentiment in ['positive', 'negative']:
                covered = words & (self.positive_indicators | self.negative_indicators | 
                                 self.negation_words | self.intensity_words)
                if not covered:
                    uncovered_sentiment_words.update(words)
        
        coverage_stats = {
            'total_samples': total_samples,
            'negation_coverage': samples_with_negation / total_samples,
            'intensity_coverage': samples_with_intensity / total_samples,
            'contrast_coverage': samples_with_contrast / total_samples,
            'positive_indicator_coverage': samples_with_positive / total_samples,
            'negative_indicator_coverage': samples_with_negative / total_samples,
            'lexicon_sizes': {
                'negation': len(self.negation_words),
                'intensity': len(self.intensity_words),
                'contrast': len(self.contrast_words),
                'positive_indicators': len(self.positive_indicators),
                'negative_indicators': len(self.negative_indicators)
            },
            'top_uncovered_words': uncovered_sentiment_words.most_common(50)
        }
        
        return coverage_stats
    
    def print_coverage_report(self, coverage_stats: Dict):
        """
        Print coverage report.
        
        Args:
            coverage_stats: Coverage statistics
        """
        print("\n" + "="*80)
        print("LEXICON COVERAGE REPORT")
        print("="*80)
        print(f"Total samples: {coverage_stats['total_samples']}")
        print(f"\nLexicon sizes:")
        for category, size in coverage_stats['lexicon_sizes'].items():
            print(f"  {category}: {size} words")
        
        print(f"\nCoverage on dataset:")
        print(f"  Negation words: {coverage_stats['negation_coverage']:.2%}")
        print(f"  Intensity words: {coverage_stats['intensity_coverage']:.2%}")
        print(f"  Contrast words: {coverage_stats['contrast_coverage']:.2%}")
        print(f"  Positive indicators: {coverage_stats['positive_indicator_coverage']:.2%}")
        print(f"  Negative indicators: {coverage_stats['negative_indicator_coverage']:.2%}")
        
        # Check if target is met
        target_coverage = 0.85
        min_coverage = min([
            coverage_stats['negation_coverage'],
            coverage_stats['intensity_coverage'],
            coverage_stats['positive_indicator_coverage'],
            coverage_stats['negative_indicator_coverage']
        ])
        
        print(f"\nMinimum coverage: {min_coverage:.2%}")
        if min_coverage >= target_coverage:
            print(f"✓ Target coverage ({target_coverage:.0%}) met!")
        else:
            print(f"✗ Target coverage ({target_coverage:.0%}) not met")
        
        if coverage_stats['top_uncovered_words']:
            print(f"\nTop potentially missing sentiment words:")
            for word, count in coverage_stats['top_uncovered_words'][:20]:
                print(f"  {word}: {count}")
        
        print("="*80 + "\n")
    
    def visualize_coverage(self, coverage_stats: Dict, save_path: str = None):
        """
        Visualize lexicon coverage.
        
        Args:
            coverage_stats: Coverage statistics
            save_path: Path to save plot
        """
        categories = [
            'Negation',
            'Intensity',
            'Contrast',
            'Positive',
            'Negative'
        ]
        
        coverages = [
            coverage_stats['negation_coverage'],
            coverage_stats['intensity_coverage'],
            coverage_stats['contrast_coverage'],
            coverage_stats['positive_indicator_coverage'],
            coverage_stats['negative_indicator_coverage']
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories, coverages, color=['#3498db', '#e74c3c', '#95a5a6', '#2ecc71', '#f39c12'])
        plt.ylabel('Coverage (%)')
        plt.title('Lexicon Coverage on Dataset')
        plt.ylim(0, 1.0)
        plt.axhline(y=0.85, color='r', linestyle='--', label='Target (85%)')
        plt.legend()
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def suggest_additions(self, dataset: List[Dict], min_frequency: int = 3) -> Dict[str, List[str]]:
        """
        Suggest words to add to lexicons.
        
        Args:
            dataset: List of data samples
            min_frequency: Minimum frequency for suggestions
            
        Returns:
            Dictionary of suggested additions by category
        """
        # Common sentiment words in Vietnamese
        common_positive = {'tốt', 'đẹp', 'hay', 'ngon', 'tuyệt', 'ổn', 'ok', 'đáng', 'chất lượng'}
        common_negative = {'xấu', 'kém', 'tệ', 'dở', 'thất vọng', 'chán', 'tồi'}
        common_negation = {'không', 'chẳng', 'chưa', 'ko', 'k'}
        
        # Find words in sentiment contexts
        word_sentiments = Counter()
        
        for sample in dataset:
            text = sample.get('processed_text', sample['text']).lower()
            sentiment = sample.get('overall_sentiment', '')
            words = set(text.split())
            
            if sentiment == 'positive':
                for word in words:
                    if word not in self.positive_indicators and len(word) > 2:
                        word_sentiments[('positive', word)] += 1
            elif sentiment == 'negative':
                for word in words:
                    if word not in self.negative_indicators and len(word) > 2:
                        word_sentiments[('negative', word)] += 1
        
        # Filter by frequency
        suggestions = {
            'positive_indicators': [],
            'negative_indicators': [],
            'negation_words': []
        }
        
        for (sentiment_type, word), count in word_sentiments.items():
            if count >= min_frequency:
                if sentiment_type == 'positive' and word not in self.positive_indicators:
                    suggestions['positive_indicators'].append((word, count))
                elif sentiment_type == 'negative' and word not in self.negative_indicators:
                    suggestions['negative_indicators'].append((word, count))
        
        # Sort by frequency
        for key in suggestions:
            suggestions[key] = sorted(suggestions[key], key=lambda x: x[1], reverse=True)
        
        return suggestions


def validate_lexicon(config_path: str, dataset_path: str) -> Dict:
    """
    Validate lexicon coverage on dataset.
    
    Args:
        config_path: Path to configuration file
        dataset_path: Path to dataset file
        
    Returns:
        Coverage statistics
    """
    from .utils import load_config, load_json
    
    config = load_config(config_path)
    dataset = load_json(dataset_path)
    
    validator = LexiconValidator(config)
    coverage_stats = validator.analyze_coverage(dataset)
    validator.print_coverage_report(coverage_stats)
    
    return coverage_stats
