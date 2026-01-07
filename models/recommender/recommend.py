import pandas as pd
import numpy as np
import random
from typing import Dict, List

class MockProductRecommender:
    def __init__(self):
        # Mock product database
        self.products = self._create_mock_products()
        self.skin_type_mapping = {
            1: "dry",
            2: "oily", 
            3: "combination",
            4: "normal",
            5: "sensitive",
            6: "mature"
        }
        
    def _create_mock_products(self):
        """Create a mock product database"""
        brands = ["Maybelline", "L'Oreal", "Fenty", "MAC", "NARS", "Estee Lauder", "Clinique"]
        product_types = ["foundation", "concealer", "powder", "serum", "moisturizer", "cleanser"]
        
        products = []
        for i in range(100):
            product = {
                "product_id": i + 1,
                "name": f"{random.choice(brands)} {random.choice(product_types)} {random.choice(['Pro', 'Matte', 'Hydrating', 'Radiant'])}",
                "brand": random.choice(brands),
                "product_type": random.choice(product_types),
                "price": round(random.uniform(10, 100), 2),
                "skin_tone_range": f"{random.randint(1, 5)}-{random.randint(6, 10)}",
                "fitzpatrick_range": f"{random.randint(1, 3)}-{random.randint(4, 6)}",
                "skin_type": random.choice(["dry", "oily", "combination", "normal", "sensitive"]),
                "concerns": random.sample(["acne", "wrinkles", "dark spots", "redness", "pores"], 2),
                "rating": round(random.uniform(3.5, 5.0), 1)
            }
            products.append(product)
        
        return pd.DataFrame(products)
    
    def recommend(self, skin_analysis: Dict, budget: float = None, n_recommendations: int = 5) -> List[Dict]:
        """
        Recommend products based on skin analysis
        
        Args:
            skin_analysis: Dictionary with 'monk_skin_tone' and 'fitzpatrick' predictions
            budget: Maximum price (optional)
            n_recommendations: Number of recommendations to return
        """
        monk_tone = skin_analysis.get("monk_skin_tone", {}).get("prediction", 5)
        fitzpatrick = skin_analysis.get("fitzpatrick", {}).get("prediction", 3)
        
        # Filter products that match skin tone
        def matches_skin_tone(row):
            try:
                low, high = map(int, row["skin_tone_range"].split("-"))
                return low <= monk_tone <= high
            except:
                return True
        
        def matches_fitzpatrick(row):
            try:
                low, high = map(int, row["fitzpatrick_range"].split("-"))
                return low <= fitzpatrick <= high
            except:
                return True
        
        # Apply filters
        filtered = self.products[
            self.products.apply(matches_skin_tone, axis=1) &
            self.products.apply(matches_fitzpatrick, axis=1)
        ]
        
        # Filter by budget if provided
        if budget:
            filtered = filtered[filtered["price"] <= budget]
        
        # If no matches, return random products
        if len(filtered) == 0:
            filtered = self.products
        
        # Sort by rating and get top recommendations
        recommendations = filtered.sort_values("rating", ascending=False).head(n_recommendations)
        
        # Add match score
        recommendations = recommendations.copy()
        recommendations["match_score"] = recommendations.apply(
            lambda row: self._calculate_match_score(row, monk_tone, fitzpatrick), 
            axis=1
        )
        
        return recommendations.to_dict("records")
    
    def _calculate_match_score(self, product, monk_tone, fitzpatrick):
        """Calculate how well a product matches the user's skin"""
        score = 0
        
        # Check skin tone match
        try:
            low, high = map(int, product["skin_tone_range"].split("-"))
            if low <= monk_tone <= high:
                score += 0.5
        except:
            pass
        
        # Check Fitzpatrick match
        try:
            low, high = map(int, product["fitzpatrick_range"].split("-"))
            if low <= fitzpatrick <= high:
                score += 0.3
        except:
            pass
        
        # Higher rating products get bonus
        score += product["rating"] / 20
        
        return round(score, 2)


def test_recommender():
    """Test the recommendation system"""
    # Create analyzer and get skin analysis
    from models.vision.predict import SkinAnalyzer
    
    print("Testing Recommendation System...")
    print("="*60)
    
    # Get skin analysis (you can use a real image or mock data)
    analyzer = SkinAnalyzer()
    
    # Try to analyze an image if exists
    test_image = "dataset/images/-10814178347722472.png"
    if os.path.exists(test_image):
        print(f"Analyzing image: {test_image}")
        skin_analysis = analyzer.analyze(test_image)
    else:
        # Use mock analysis for testing
        print("Using mock skin analysis for testing...")
        skin_analysis = {
            "monk_skin_tone": {"prediction": 4, "confidence": 0.62},
            "fitzpatrick": {"prediction": 5, "confidence": 0.28}
        }
    
    print(f"\nSkin Analysis:")
    print(f"  Monk Skin Tone: {skin_analysis['monk_skin_tone']['prediction']}")
    print(f"  Fitzpatrick: {skin_analysis['fitzpatrick']['prediction']}")
    
    # Get recommendations
    recommender = MockProductRecommender()
    recommendations = recommender.recommend(
        skin_analysis=skin_analysis,
        budget=50.0,  # $50 budget
        n_recommendations=5
    )
    
    print(f"\nTop 5 Product Recommendations (Budget: $50):")
    print("="*60)
    for i, product in enumerate(recommendations, 1):
        print(f"\n{i}. {product['name']}")
        print(f"   Brand: {product['brand']}")
        print(f"   Type: {product['product_type']}")
        print(f"   Price: ${product['price']:.2f}")
        print(f"   Rating: {product['rating']}/5")
        print(f"   Match Score: {product['match_score']}/1.0")
        print(f"   Skin Tone Range: {product['skin_tone_range']}")
        print(f"   Suitable for: {product['skin_type']} skin")
    
    return recommendations


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    test_recommender()