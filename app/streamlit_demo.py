#!/usr/bin/env python3
"""Streamlit demo for Vietnamese ABSA."""

import streamlit as st
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import create_pipeline


# Page config
st.set_page_config(
    page_title="Vietnamese ABSA Demo",
    page_icon="🇻🇳",
    layout="wide"
)


@st.cache_resource
def load_pipeline():
    """Load ABSA pipeline (cached)."""
    try:
        pipeline = create_pipeline(
            stage1_model_path="models/stage1_aspect_extraction",
            stage2_model_path="models/stage2_sentiment",
            config_path="config/config.yaml"
        )
        return pipeline, None
    except Exception as e:
        return None, str(e)


def get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for sentiment."""
    emoji_map = {
        'positive': '🟢',
        'negative': '🔴',
        'neutral': '⚪',
        'mixed': '🟡'
    }
    return emoji_map.get(sentiment, '⚪')


def get_sentiment_color(sentiment: str) -> str:
    """Get color for sentiment."""
    color_map = {
        'positive': '#4CAF50',
        'negative': '#F44336',
        'neutral': '#9E9E9E',
        'mixed': '#FFC107'
    }
    return color_map.get(sentiment, '#9E9E9E')


def main():
    # Title
    st.title("🇻🇳 Vietnamese Aspect-Based Sentiment Analysis")
    st.markdown("*Phân tích cảm xúc theo khía cạnh cho văn bản tiếng Việt*")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This demo showcases Vietnamese ABSA using PhoBERT:
        
        **Features:**
        - Aspect extraction (NER)
        - Sentiment classification
        - Support for negation and intensity
        - Emoji understanding
        
        **Model:**
        - Stage 1: PhoBERT + NER
        - Stage 2: PhoBERT + Classification
        """)
        
        st.header("📝 Examples")
        examples = [
            "Điện thoại này màn hình đẹp nhưng pin yếu quá",
            "Máy chạy rất mượt, hiệu năng tốt",
            "Camera chụp ảnh đẹp, màu sắc sống động",
            "Thiết kế đẹp mắt, nhưng giá hơi cao"
        ]
        
        for example in examples:
            if st.button(example, key=example):
                st.session_state.example_text = example
    
    # Load pipeline
    with st.spinner("Loading model..."):
        pipeline, error = load_pipeline()
    
    if error:
        st.error(f"Error loading model: {error}")
        st.info("Please ensure models are trained and saved in the correct directories.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Input text
    default_text = st.session_state.get('example_text', '')
    text_input = st.text_area(
        "Enter Vietnamese text:",
        value=default_text,
        height=100,
        placeholder="Nhập văn bản tiếng Việt tại đây..."
    )
    
    # Predict button
    if st.button("🔍 Analyze", type="primary"):
        if not text_input or not text_input.strip():
            st.warning("Please enter some text!")
            return
        
        # Run prediction
        with st.spinner("Analyzing..."):
            try:
                result = pipeline.predict(text_input)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return
        
        # Display results
        st.markdown("---")
        st.header("📊 Results")
        
        # Overall sentiment
        overall_sentiment = result['overall_sentiment']
        overall_emoji = get_sentiment_emoji(overall_sentiment)
        overall_color = get_sentiment_color(overall_sentiment)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric(
                label="Overall Sentiment",
                value=overall_sentiment.upper()
            )
        with col2:
            st.markdown(
                f"<h2 style='color: {overall_color};'>{overall_emoji} {overall_sentiment.upper()}</h2>",
                unsafe_allow_html=True
            )
        
        # Aspect-level results
        st.subheader("🎯 Aspect-Level Analysis")
        
        if result['aspects']:
            for i, aspect in enumerate(result['aspects'], 1):
                aspect_name = aspect['aspect']
                sentiment = aspect['sentiment']
                confidence = aspect['confidence']
                
                emoji = get_sentiment_emoji(sentiment)
                color = get_sentiment_color(sentiment)
                
                with st.container():
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown(f"**{i}. Aspect:** `{aspect_name}`")
                    
                    with col2:
                        st.markdown(
                            f"{emoji} **Sentiment:** <span style='color: {color};'>{sentiment.upper()}</span>",
                            unsafe_allow_html=True
                        )
                    
                    with col3:
                        st.markdown(f"**Confidence:** {confidence:.2%}")
                    
                    st.progress(confidence)
                    st.markdown("---")
        else:
            st.info("No aspects detected in the text.")
        
        # Raw JSON output
        with st.expander("🔍 View Raw JSON"):
            st.json(result)
        
        # Preprocessed text
        with st.expander("📝 View Preprocessed Text"):
            st.code(result['processed_text'], language='text')


if __name__ == "__main__":
    main()
