#!/usr/bin/env python3
"""Download and prepare Vietnamese ABSA datasets."""

import argparse
import json
import os
from pathlib import Path
import sys


def create_sample_data():
    """Create sample Vietnamese ABSA data for demonstration."""
    sample_data = [
        {
            "text": "Điện thoại này màn hình đẹp nhưng pin yếu quá",
            "aspects": [
                {"term": "màn hình", "span": [18, 27], "sentiment": "positive"},
                {"term": "pin", "span": [35, 38], "sentiment": "negative"}
            ],
            "overall_sentiment": "mixed"
        },
        {
            "text": "Máy chạy rất mượt, hiệu năng tốt",
            "aspects": [
                {"term": "hiệu năng", "span": [21, 30], "sentiment": "positive"}
            ],
            "overall_sentiment": "positive"
        },
        {
            "text": "Camera chụp ảnh đẹp, màu sắc sống động",
            "aspects": [
                {"term": "Camera", "span": [0, 6], "sentiment": "positive"}
            ],
            "overall_sentiment": "positive"
        },
        {
            "text": "Sản phẩm tệ, chất lượng kém, không đáng tiền",
            "aspects": [
                {"term": "chất lượng", "span": [13, 23], "sentiment": "negative"}
            ],
            "overall_sentiment": "negative"
        },
        {
            "text": "Thiết kế đẹp mắt, nhưng giá hơi cao",
            "aspects": [
                {"term": "Thiết kế", "span": [0, 8], "sentiment": "positive"},
                {"term": "giá", "span": [24, 27], "sentiment": "negative"}
            ],
            "overall_sentiment": "mixed"
        },
        {
            "text": "Bàn phím gõ tốt, âm thanh rõ ràng",
            "aspects": [
                {"term": "Bàn phím", "span": [0, 8], "sentiment": "positive"},
                {"term": "âm thanh", "span": [17, 25], "sentiment": "positive"}
            ],
            "overall_sentiment": "positive"
        },
        {
            "text": "Pin trâu, sử dụng cả ngày không lo hết pin",
            "aspects": [
                {"term": "Pin", "span": [0, 3], "sentiment": "positive"}
            ],
            "overall_sentiment": "positive"
        },
        {
            "text": "Sản phẩm bình thường, không có gì đặc biệt",
            "aspects": [],
            "overall_sentiment": "neutral"
        },
        {
            "text": "Máy nóng quá, dùng 1 tiếng là nóng rát tay",
            "aspects": [
                {"term": "Máy", "span": [0, 3], "sentiment": "negative"}
            ],
            "overall_sentiment": "negative"
        },
        {
            "text": "Hàng chính hãng, bảo hành tốt, giá hợp lý",
            "aspects": [
                {"term": "bảo hành", "span": [17, 25], "sentiment": "positive"},
                {"term": "giá", "span": [31, 34], "sentiment": "positive"}
            ],
            "overall_sentiment": "positive"
        }
    ]
    
    return sample_data


def download_datasets(output_dir: str = "data/raw"):
    """
    Download Vietnamese ABSA datasets.
    
    Note: This is a placeholder. In production, integrate with:
    - UIT-VSFC (Vietnamese Students' Feedback Corpus)
    - ViHSD (Vietnamese Hate Speech Detection) - can be adapted
    - VLSP sentiment datasets
    - Or use web scraping from Vietnamese review sites
    
    Args:
        output_dir: Output directory for datasets
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("DATASET DOWNLOAD")
    print("="*80)
    print("\nNote: This creates sample data for demonstration.")
    print("For production, integrate with actual Vietnamese datasets:")
    print("  - UIT-VSFC (Vietnamese Students' Feedback Corpus)")
    print("  - VLSP Sentiment Analysis datasets")
    print("  - Web scraping from Vietnamese e-commerce sites")
    print()
    
    # Create sample data
    print("Creating sample Vietnamese ABSA data...")
    sample_data = create_sample_data()
    
    # Save sample data
    output_file = output_path / "sample_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Sample data saved to {output_file}")
    print(f"  Total samples: {len(sample_data)}")
    
    print("\n" + "="*80)
    print("DATASET DOWNLOAD COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Validate data: python scripts/validate_data.py --data {output_file}")
    print(f"2. Split data: python scripts/split_data.py --input {output_file}")
    print(f"3. Preprocess data: python scripts/preprocess_data.py --input data/processed/train.json --output data/processed/train.json")
    print()


def main():
    parser = argparse.ArgumentParser(description='Download Vietnamese ABSA datasets')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                       help='Output directory for datasets')
    
    args = parser.parse_args()
    
    download_datasets(args.output_dir)


if __name__ == '__main__':
    main()
