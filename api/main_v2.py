from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vision.predict_v2 import SkinAnalyzerV2
from models.recommender.recommend import MockProductRecommender

app = FastAPI(title="Beauty Recommendation API V2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize improved models
skin_analyzer = None
product_recommender = None

def get_skin_analyzer():
    global skin_analyzer
    if skin_analyzer is None:
        skin_analyzer = SkinAnalyzerV2("models/vision/beauty_vision_model_v2.pt")
    return skin_analyzer

def get_product_recommender():
    global product_recommender
    if product_recommender is None:
        product_recommender = MockProductRecommender()
    return product_recommender

@app.get("/")
async def root():
    return {
        "message": "Beauty Recommendation API V2",
        "model_version": "ImprovedBeautyVisionModel v2",
        "accuracy": "~38% (improved from ~30%)",
        "endpoints": {
            "GET /": "This message",
            "POST /analyze": "Analyze skin from image (using improved model)",
            "POST /recommend": "Get product recommendations",
            "GET /products": "Browse all products",
            "GET /health": "Health check",
            "GET /model-info": "Get model information"
        }
    }

@app.get("/model-info")
async def model_info():
    """Get information about the current model"""
    analyzer = get_skin_analyzer()
    return {
        "model_version": "v2",
        "model_type": "ImprovedBeautyVisionModel (ResNet34)",
        "accuracy": {
            "skin_tone": "~38%",
            "fitzpatrick": "~38%"
        },
        "improvement": "~8% over previous model",
        "features": ["Data augmentation", "Class weights", "Learning rate scheduling", "Gradient clipping"]
    }

@app.post("/analyze")
async def analyze_skin(image: UploadFile = File(...)):
    """Analyze skin from uploaded image using improved model"""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, image.filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        analyzer = get_skin_analyzer()
        result = analyzer.analyze(temp_path)
        
        os.remove(temp_path)
        
        return {
            "success": True,
            "message": "Analysis complete (using improved model v2)",
            "model_version": "v2",
            "analysis": result
        }
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/recommend")
async def get_recommendations(
    monk_skin_tone: int,
    fitzpatrick: int,
    budget: float = None,
    n_recommendations: int = 5
):
    try:
        skin_analysis = {
            "monk_skin_tone": {"prediction": monk_skin_tone},
            "fitzpatrick": {"prediction": fitzpatrick}
        }
        
        recommender = get_product_recommender()
        recommendations = recommender.recommend(
            skin_analysis=skin_analysis,
            budget=budget,
            n_recommendations=n_recommendations
        )
        
        return {
            "success": True,
            "parameters": {
                "monk_skin_tone": monk_skin_tone,
                "fitzpatrick": fitzpatrick,
                "budget": budget,
                "n_recommendations": n_recommendations
            },
            "recommendations": recommendations,
            "total_recommendations": len(recommendations)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Recommendation failed: {str(e)}")

@app.get("/products")
async def browse_products(
    page: int = 1,
    page_size: int = 10,
    min_price: float = None,
    max_price: float = None,
    brand: str = None
):
    try:
        recommender = get_product_recommender()
        products = recommender.products.copy()
        
        if min_price is not None:
            products = products[products["price"] >= min_price]
        if max_price is not None:
            products = products[products["price"] <= max_price]
        if brand:
            products = products[products["brand"].str.contains(brand, case=False, na=False)]
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_products = products.iloc[start_idx:end_idx]
        
        return {
            "success": True,
            "page": page,
            "page_size": page_size,
            "total_products": len(products),
            "total_pages": (len(products) + page_size - 1) // page_size,
            "products": paginated_products.to_dict("records")
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to browse products: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "beauty-recommendation-api-v2"}

if __name__ == "__main__":
    print("Starting Beauty Recommendation API V2 on http://localhost:8000")
    print("Using improved model with ~38% accuracy")
    uvicorn.run(app, host="0.0.0.0", port=8000)