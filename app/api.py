#!/usr/bin/env python3
"""FastAPI server for Vietnamese ABSA."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import create_pipeline


# Request/Response models
class PredictionRequest(BaseModel):
    text: str


class AspectResult(BaseModel):
    aspect: str
    sentiment: str
    confidence: float


class PredictionResponse(BaseModel):
    text: str
    processed_text: str
    aspects: List[AspectResult]
    overall_sentiment: str


# Create FastAPI app
app = FastAPI(
    title="Vietnamese ABSA API",
    description="Aspect-Based Sentiment Analysis for Vietnamese text",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline (loaded at startup)
pipeline = None


@app.on_event("startup")
async def load_model():
    """Load ABSA pipeline at startup."""
    global pipeline
    
    print("Loading ABSA pipeline...")
    try:
        pipeline = create_pipeline(
            stage1_model_path="models/stage1_aspect_extraction",
            stage2_model_path="models/stage2_sentiment",
            config_path="config/config.yaml"
        )
        print("Pipeline loaded successfully!")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Please ensure models are trained and saved in the correct directories.")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Vietnamese ABSA API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if pipeline is None:
        return {"status": "unhealthy", "message": "Pipeline not loaded"}
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict aspects and sentiments for Vietnamese text.
    
    Args:
        request: Request with text field
        
    Returns:
        Prediction results with aspects and sentiments
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Run prediction
        result = pipeline.predict(request.text)
        
        # Format response
        aspects = [
            AspectResult(
                aspect=a['aspect'],
                sentiment=a['sentiment'],
                confidence=a['confidence']
            )
            for a in result['aspects']
        ]
        
        response = PredictionResponse(
            text=result['text'],
            processed_text=result['processed_text'],
            aspects=aspects,
            overall_sentiment=result['overall_sentiment']
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(texts: List[str]):
    """
    Batch prediction for multiple texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        List of prediction results
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    try:
        results = []
        for text in texts:
            if text and text.strip():
                result = pipeline.predict(text)
                results.append(result)
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


def main():
    """Run FastAPI server."""
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )


if __name__ == "__main__":
    main()
