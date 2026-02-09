# Vietnamese ABSA Quick Start Guide

Get started with Vietnamese Aspect-Based Sentiment Analysis in minutes!

## 🚀 Quick Install

```bash
git clone https://github.com/WynT08/NLP.git
cd NLP
pip install -r requirements.txt
```

## 📝 Create Sample Data

```bash
python scripts/download_datasets.py
```

This creates `data/raw/sample_data.json` with 10 Vietnamese ABSA examples.

## 🔧 Prepare Data

```bash
# Validate annotations
python scripts/validate_data.py --data data/raw/sample_data.json

# Split into train/val/test
python scripts/split_data.py \
    --input data/raw/sample_data.json \
    --output-dir data/processed

# Preprocess all splits
for split in train val test; do
    python scripts/preprocess_data.py \
        --input data/processed/${split}.json \
        --output data/processed/${split}.json
done
```

## 🎓 Train Models

### Stage 1: Aspect Extraction

```bash
python scripts/train_stage1.py
```

Expected output:
- Training completes in ~3-5 hours on RTX 4050
- F1 Score: ≥0.85
- Model saved to `models/stage1_aspect_extraction/`

### Stage 2: Sentiment Classification

```bash
python scripts/train_stage2.py
```

Expected output:
- Training completes in ~3-5 hours on RTX 4050
- Macro-F1: ≥0.90
- Model saved to `models/stage2_sentiment/`

## 🧪 Evaluate

```bash
python scripts/evaluate.py \
    --stage1-model models/stage1_aspect_extraction \
    --stage2-model models/stage2_sentiment \
    --test-data data/processed/test.json \
    --output evaluation_results.json
```

## 💻 Use in Python

```python
from src.inference import create_pipeline

# Load pipeline
pipeline = create_pipeline(
    stage1_model_path="models/stage1_aspect_extraction",
    stage2_model_path="models/stage2_sentiment"
)

# Analyze text
text = "Điện thoại này màn hình đẹp nhưng pin yếu quá"
result = pipeline.predict(text)

# Print results
print(f"Overall: {result['overall_sentiment']}")
for aspect in result['aspects']:
    print(f"- {aspect['aspect']}: {aspect['sentiment']} ({aspect['confidence']:.0%})")
```

Output:
```
Overall: mixed
- màn hình: positive (92%)
- pin: negative (89%)
```

## 🌐 Deploy Demo

### Option 1: Streamlit (Interactive UI)

```bash
streamlit run app/streamlit_demo.py
```

Visit `http://localhost:8501`

### Option 2: FastAPI (REST API)

```bash
cd app
uvicorn api:app --host 0.0.0.0 --port 8000
```

Test:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Điện thoại này màn hình đẹp"}'
```

## 📊 Optional: Data Augmentation

Increase training data by 30%:

```bash
python scripts/augment_data.py \
    --input data/processed/train.json \
    --output data/augmented/train_augmented.json \
    --factor 1.3
```

Then retrain Stage 1 and Stage 2 with augmented data.

## ✅ Validation Checklist

- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Created or obtained Vietnamese ABSA data
- [ ] Validated data format
- [ ] Preprocessed data (negation/intensity masks)
- [ ] Trained Stage 1 model (F1 ≥0.85)
- [ ] Trained Stage 2 model (Macro-F1 ≥0.90)
- [ ] Evaluated on test set
- [ ] Deployed demo (Streamlit or FastAPI)

## 🆘 Troubleshooting

### Issue: CUDA out of memory

**Solution**: Reduce batch size in config files:
```yaml
# config/training_stage1.yaml or training_stage2.yaml
training:
  per_device_train_batch_size: 8  # Reduce from 16
  gradient_accumulation_steps: 4  # Increase to compensate
```

### Issue: Low performance

**Solutions**:
1. **More training data**: Aim for 1000+ samples
2. **Data augmentation**: Use `augment_data.py`
3. **Validate lexicon coverage**: Run `validate_lexicon.py`
4. **Increase epochs**: Edit training configs

### Issue: Models not found

**Solution**: Ensure you've trained both stages:
```bash
python scripts/train_stage1.py  # Creates models/stage1_aspect_extraction/
python scripts/train_stage2.py  # Creates models/stage2_sentiment/
```

## 📚 Next Steps

1. **Annotate more data**: Create domain-specific Vietnamese ABSA dataset
2. **Fine-tune lexicons**: Add domain-specific negation/intensity words
3. **Evaluate on domain**: Test on your specific use case
4. **Optimize for production**: Quantization, ONNX export, TensorRT

## 🎯 Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Stage 1 F1 | ≥0.85 | Aspect extraction accuracy |
| Stage 2 Macro-F1 | ≥0.90 | Sentiment classification accuracy |
| End-to-end | ≥0.80 | Aspect + sentiment correct |
| Overall | >95% | System accuracy |

## 💡 Tips

- **Start small**: Test with sample data first
- **Validate early**: Check data quality before training
- **Monitor training**: Watch for overfitting (validation loss)
- **Iterate**: Improve lexicons based on error analysis
- **Scale up**: Move to GPU for larger datasets

## 📞 Need Help?

- Check the full [README.md](README.md)
- Review [demo.ipynb](notebooks/demo.ipynb)
- Open an issue on GitHub
- Check configuration files in `config/`

---

**Ready to go!** 🎉 Start with sample data and scale up to your domain.
