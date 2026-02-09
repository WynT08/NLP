# Vietnamese Aspect-Based Sentiment Analysis (ABSA)

A complete, production-ready Vietnamese Aspect-Based Sentiment Analysis system using PhoBERT. This two-stage pipeline extracts aspects from Vietnamese text and classifies sentiment for each aspect.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **Two-Stage Pipeline**: Aspect extraction (NER) → Sentiment classification
- **PhoBERT-powered**: Uses state-of-the-art Vietnamese language model
- **Comprehensive Preprocessing**: 
  - Vietnamese text normalization
  - Slang/teencode handling (ko→không, wá→quá)
  - Negation detection (50+ negation words)
  - Intensity detection (100+ intensity words)
  - Emoji understanding (80+ sentiment emojis)
- **Data Augmentation**: Back-translation, contextual replacement, random operations
- **Production-Ready**: FastAPI server + Streamlit demo
- **GPU-Optimized**: FP16 training, gradient accumulation for RTX GPUs
- **High Accuracy**: Target >95% overall, >85% F1 on aspects

## 📁 Project Structure

```
vietnamese-absa/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── config/
│   ├── config.yaml                  # Main configuration
│   ├── training_stage1.yaml         # Stage 1 training config
│   └── training_stage2.yaml         # Stage 2 training config
├── data/
│   ├── raw/                         # Raw annotated data
│   ├── processed/                   # Preprocessed data
│   └── augmented/                   # Augmented training data
├── models/
│   ├── stage1_aspect_extraction/    # Trained NER model
│   └── stage2_sentiment/            # Trained sentiment model
├── src/
│   ├── __init__.py
│   ├── preprocessing.py             # Vietnamese text preprocessing
│   ├── data_augmentation.py         # Data augmentation
│   ├── dataset.py                   # PyTorch datasets
│   ├── models.py                    # Model architectures
│   ├── training.py                  # Training utilities
│   ├── inference.py                 # ABSA pipeline
│   ├── evaluation.py                # Evaluation metrics
│   ├── lexicon_validator.py         # Lexicon coverage analysis
│   └── utils.py                     # Helper functions
├── scripts/
│   ├── download_datasets.py         # Download/create sample data
│   ├── prepare_data.py              # Prepare raw data
│   ├── validate_data.py             # Validate annotations
│   ├── split_data.py                # Train/val/test split
│   ├── preprocess_data.py           # Preprocess text
│   ├── augment_data.py              # Augment training data
│   ├── validate_lexicon.py          # Validate lexicon coverage
│   ├── train_stage1.py              # Train aspect extraction
│   ├── train_stage2.py              # Train sentiment classification
│   └── evaluate.py                  # Evaluate pipeline
├── app/
│   ├── api.py                       # FastAPI server
│   └── streamlit_demo.py            # Interactive demo
├── notebooks/
│   └── demo.ipynb                   # Jupyter demo notebook
└── tests/
    ├── test_preprocessing.py
    ├── test_models.py
    └── test_inference.py
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/WynT08/NLP.git
cd NLP

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Quick Demo

```python
from src.inference import create_pipeline

# Load trained pipeline
pipeline = create_pipeline(
    stage1_model_path="models/stage1_aspect_extraction",
    stage2_model_path="models/stage2_sentiment"
)

# Analyze Vietnamese text
text = "Điện thoại này màn hình đẹp nhưng pin yếu quá"
result = pipeline.predict(text)

print(f"Overall: {result['overall_sentiment']}")
for aspect in result['aspects']:
    print(f"- {aspect['aspect']}: {aspect['sentiment']} ({aspect['confidence']:.2%})")
```

## 📊 Data Format

### Input Format (Annotation)

```json
{
  "text": "Điện thoại này màn hình đẹp nhưng pin yếu quá",
  "aspects": [
    {"term": "màn hình", "span": [18, 27], "sentiment": "positive"},
    {"term": "pin", "span": [35, 38], "sentiment": "negative"}
  ],
  "overall_sentiment": "mixed"
}
```

### Output Format (Prediction)

```json
{
  "text": "Điện thoại này màn hình đẹp nhưng pin yếu quá",
  "processed_text": "điện thoại này màn_hình đẹp nhưng pin yếu quá",
  "aspects": [
    {"aspect": "màn hình", "sentiment": "positive", "confidence": 0.92},
    {"aspect": "pin", "sentiment": "negative", "confidence": 0.89}
  ],
  "overall_sentiment": "mixed"
}
```

## 🔧 Full Training Pipeline

### Step 1: Prepare Data

```bash
# Download/create sample data
python scripts/download_datasets.py --output-dir data/raw

# Validate data format
python scripts/validate_data.py --data data/raw/sample_data.json

# Split into train/val/test
python scripts/split_data.py \
    --input data/raw/sample_data.json \
    --output-dir data/processed \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

### Step 2: Preprocess Data

```bash
# Preprocess all splits
for split in train val test; do
    python scripts/preprocess_data.py \
        --input data/processed/${split}.json \
        --output data/processed/${split}.json \
        --config config/config.yaml
done
```

### Step 3: Augment Training Data (Optional)

```bash
python scripts/augment_data.py \
    --input data/processed/train.json \
    --output data/augmented/train_augmented.json \
    --factor 1.3
```

### Step 4: Validate Lexicon Coverage

```bash
python scripts/validate_lexicon.py \
    --config config/config.yaml \
    --data data/processed/train.json \
    --visualize \
    --output lexicon_coverage.png
```

### Step 5: Train Stage 1 (Aspect Extraction)

```bash
python scripts/train_stage1.py --config config/training_stage1.yaml
```

**Expected Results:**
- F1 Score: ≥0.85
- Precision: ≥0.83
- Recall: ≥0.87

### Step 6: Train Stage 2 (Sentiment Classification)

```bash
python scripts/train_stage2.py --config config/training_stage2.yaml
```

**Expected Results:**
- Accuracy: ≥0.90
- Macro F1: ≥0.90
- Weighted F1: ≥0.92

### Step 7: Evaluate End-to-End

```bash
python scripts/evaluate.py \
    --stage1-model models/stage1_aspect_extraction \
    --stage2-model models/stage2_sentiment \
    --test-data data/processed/test.json \
    --output evaluation_results.json
```

## 🌐 Deployment

### FastAPI Server

```bash
# Start API server
cd app
uvicorn api:app --host 0.0.0.0 --port 8000

# Test API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Điện thoại này màn hình đẹp"}'
```

### Streamlit Demo

```bash
# Start Streamlit app
streamlit run app/streamlit_demo.py
```

Visit `http://localhost:8501` to use the interactive demo.

## ⚙️ Configuration

### Main Configuration (`config/config.yaml`)

Key settings:
- **Model**: PhoBERT backbone, max sequence length
- **Preprocessing**: Lexicons (negation, intensity, emojis)
- **Augmentation**: Back-translation, MLM, random ops
- **Device**: CUDA/CPU, FP16 settings

### Training Configurations

**Stage 1** (`config/training_stage1.yaml`):
- Epochs: 10
- Batch size: 16
- Learning rate: 2e-5
- FP16: True
- Early stopping: 3 epochs
- Metric: F1 score

**Stage 2** (`config/training_stage2.yaml`):
- Epochs: 8
- Batch size: 16
- Class weights: Enabled
- Negation/intensity features: Enabled
- Metric: Macro F1

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_preprocessing.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📈 Performance

### Hardware Requirements
- **Minimum**: 8GB RAM, NVIDIA GPU (optional)
- **Recommended**: 16GB RAM, NVIDIA RTX 4050/4060 (6GB VRAM)
- **Training Time**: 3-5 hours per stage on RTX 4050

### Performance Targets
- Stage 1 (Aspect Extraction) F1: ≥0.85
- Stage 2 (Sentiment) Macro-F1: ≥0.90
- End-to-end Aspect-level Accuracy: ≥0.80
- Overall System Accuracy: >95%

## 🔍 Model Details

### Stage 1: Aspect Extraction
- **Architecture**: PhoBERT + Linear NER head
- **Labels**: B-ASP, I-ASP, O (BIO tagging)
- **Input**: Tokenized Vietnamese text
- **Output**: Aspect term spans

### Stage 2: Sentiment Classification
- **Architecture**: PhoBERT + MLP classifier
- **Input**: [CLS] text [SEP] aspect [SEP]
- **Features**: Optional negation/intensity masks
- **Labels**: positive, negative, neutral, mixed

## 🗂️ Lexicons

The system includes comprehensive Vietnamese lexicons:

- **Negation Words**: 50+ (không, chẳng, ko, k, etc.)
- **Intensity Words**: 100+ (rất, cực, quá, siêu, etc.)
- **Contrast Words**: 20+ (nhưng, mà, tuy nhiên, etc.)
- **Sentiment Emojis**: 80+ (😊, 😢, 😐, etc.)
- **Positive/Negative Indicators**: 200+ words

Target coverage: >85% on training data

## 📚 Citation

If you use this code, please cite:

```bibtex
@software{vietnamese_absa_2024,
  title = {Vietnamese Aspect-Based Sentiment Analysis},
  author = {Vietnamese ABSA Team},
  year = {2024},
  url = {https://github.com/WynT08/NLP}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🙏 Acknowledgments

- **VinAI Research** for PhoBERT
- **underthesea** for Vietnamese NLP tools
- **Hugging Face** for transformers library

## 🔗 References

- [PhoBERT: Pre-trained language models for Vietnamese](https://github.com/VinAIResearch/PhoBERT)
- [underthesea - Vietnamese NLP Toolkit](https://github.com/undertheseanlp/underthesea)
- [Aspect-Based Sentiment Analysis](https://paperswithcode.com/task/aspect-based-sentiment-analysis)

---

**Note**: This is a research/educational project. For production use, please ensure proper data collection, annotation, and validation for your specific domain.
