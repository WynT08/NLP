#!/usr/bin/env python3
"""Validate annotated data for Vietnamese ABSA."""

import argparse
import json
from pathlib import Path
from collections import Counter


def validate_data(data_file: str) -> dict:
    """
    Validate data format and report issues.
    
    Args:
        data_file: Path to JSON data file
        
    Returns:
        Validation report dictionary
    """
    print(f"Validating data from {data_file}...")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_samples = len(data)
    issues = []
    
    # Statistics
    aspect_sentiments = Counter()
    overall_sentiments = Counter()
    aspects_per_sample = []
    
    for i, item in enumerate(data):
        # Check required fields
        if 'text' not in item:
            issues.append(f"Sample {i}: Missing 'text' field")
            continue
        
        if 'aspects' not in item:
            issues.append(f"Sample {i}: Missing 'aspects' field")
            item['aspects'] = []
        
        if 'overall_sentiment' not in item:
            issues.append(f"Sample {i}: Missing 'overall_sentiment' field")
        
        # Validate aspects
        aspects = item.get('aspects', [])
        aspects_per_sample.append(len(aspects))
        
        for j, aspect in enumerate(aspects):
            if 'term' not in aspect:
                issues.append(f"Sample {i}, Aspect {j}: Missing 'term' field")
            
            if 'sentiment' not in aspect:
                issues.append(f"Sample {i}, Aspect {j}: Missing 'sentiment' field")
            else:
                aspect_sentiments[aspect['sentiment']] += 1
            
            if 'span' not in aspect:
                issues.append(f"Sample {i}, Aspect {j}: Missing 'span' field")
            else:
                # Validate span
                span = aspect['span']
                if len(span) != 2:
                    issues.append(f"Sample {i}, Aspect {j}: Invalid span format")
                else:
                    start, end = span
                    text = item['text']
                    if start < 0 or end > len(text) or start >= end:
                        issues.append(f"Sample {i}, Aspect {j}: Invalid span indices [{start}, {end}]")
                    else:
                        # Check if term matches span
                        extracted_term = text[start:end]
                        if 'term' in aspect and aspect['term'] != extracted_term:
                            issues.append(f"Sample {i}, Aspect {j}: Term '{aspect['term']}' doesn't match span text '{extracted_term}'")
        
        # Count overall sentiment
        if 'overall_sentiment' in item:
            overall_sentiments[item['overall_sentiment']] += 1
    
    # Generate report
    report = {
        'total_samples': total_samples,
        'total_issues': len(issues),
        'issues': issues[:50],  # Show first 50 issues
        'aspect_sentiment_distribution': dict(aspect_sentiments),
        'overall_sentiment_distribution': dict(overall_sentiments),
        'avg_aspects_per_sample': sum(aspects_per_sample) / len(aspects_per_sample) if aspects_per_sample else 0,
        'min_aspects': min(aspects_per_sample) if aspects_per_sample else 0,
        'max_aspects': max(aspects_per_sample) if aspects_per_sample else 0
    }
    
    return report


def print_validation_report(report: dict):
    """Print validation report."""
    print("\n" + "="*80)
    print("DATA VALIDATION REPORT")
    print("="*80)
    print(f"Total samples: {report['total_samples']}")
    print(f"Total issues found: {report['total_issues']}")
    
    print(f"\nAspect sentiment distribution:")
    for sentiment, count in report['aspect_sentiment_distribution'].items():
        print(f"  {sentiment}: {count}")
    
    print(f"\nOverall sentiment distribution:")
    for sentiment, count in report['overall_sentiment_distribution'].items():
        print(f"  {sentiment}: {count}")
    
    print(f"\nAspects per sample:")
    print(f"  Average: {report['avg_aspects_per_sample']:.2f}")
    print(f"  Min: {report['min_aspects']}")
    print(f"  Max: {report['max_aspects']}")
    
    # Check class imbalance
    aspect_sentiments = report['aspect_sentiment_distribution']
    if aspect_sentiments:
        max_count = max(aspect_sentiments.values())
        min_count = min(aspect_sentiments.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 3:
            print("  ⚠ Warning: Significant class imbalance detected (ratio > 3)")
            print("  Consider using class weights or data augmentation")
    
    if report['issues']:
        print(f"\nFirst {min(len(report['issues']), 50)} issues:")
        for issue in report['issues'][:50]:
            print(f"  - {issue}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Validate Vietnamese ABSA data')
    parser.add_argument('--data', type=str, required=True, help='Path to JSON data file')
    parser.add_argument('--output', type=str, help='Path to save validation report (JSON)')
    
    args = parser.parse_args()
    
    report = validate_data(args.data)
    print_validation_report(report)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"Validation report saved to {args.output}")


if __name__ == '__main__':
    main()
