import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.vision.model import BeautyVisionModel

class SkinAnalyzer:
    def __init__(self, model_path="models/vision/beauty_vision_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = BeautyVisionModel().to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        self.model.eval()
        
        # Transform for inference
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def analyze(self, image_path):
        """Analyze a skin image and return predictions"""
        # Load image
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": f"Cannot load image: {e}"}
        
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        # Convert to probabilities
        skin_tone_probs = F.softmax(outputs["skin_tone"], dim=1)
        fitzpatrick_probs = F.softmax(outputs["fitzpatrick"], dim=1)
        
        # Get predictions
        skin_tone_pred = torch.argmax(skin_tone_probs, dim=1).item() + 1
        fitzpatrick_pred = torch.argmax(fitzpatrick_probs, dim=1).item() + 1
        
        # Get top 3 predictions for each (convert tensors to Python ints)
        skin_tone_top3 = torch.topk(skin_tone_probs, 3, dim=1)
        fitzpatrick_top3 = torch.topk(fitzpatrick_probs, 3, dim=1)
        
        return {
            "monk_skin_tone": {
                "prediction": skin_tone_pred,
                "confidence": float(skin_tone_probs.max().item()),
                "top_3": [
                    {"class": int(idx.item()) + 1, "probability": float(prob.item())}
                    for prob, idx in zip(skin_tone_top3.values[0], skin_tone_top3.indices[0])
                ]
            },
            "fitzpatrick": {
                "prediction": fitzpatrick_pred,
                "confidence": float(fitzpatrick_probs.max().item()),
                "top_3": [
                    {"class": int(idx.item()) + 1, "probability": float(prob.item())}
                    for prob, idx in zip(fitzpatrick_top3.values[0], fitzpatrick_top3.indices[0])
                ]
            }
        }
    
    def analyze_and_print(self, image_path):
        """Analyze and print formatted results"""
        result = self.analyze(image_path)
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        print("\n" + "="*60)
        print("SKIN ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\nMonk Skin Tone Scale (1-10):")
        print(f"  Prediction: {result['monk_skin_tone']['prediction']}")
        print(f"  Confidence: {result['monk_skin_tone']['confidence']:.2%}")
        print(f"  Top 3 Predictions:")
        for pred in result['monk_skin_tone']['top_3']:
            print(f"    - Class {pred['class']}: {pred['probability']:.2%}")
        
        print(f"\nFitzpatrick Scale (1-6):")
        print(f"  Prediction: {result['fitzpatrick']['prediction']}")
        print(f"  Confidence: {result['fitzpatrick']['confidence']:.2%}")
        print(f"  Top 3 Predictions:")
        for pred in result['fitzpatrick']['top_3']:
            print(f"    - Class {pred['class']}: {pred['probability']:.2%}")
        
        return result


def test_with_sample_image():
    """Test the analyzer with a sample image"""
    analyzer = SkinAnalyzer()
    
    # Try to find a test image
    test_images = [
        "test_image.jpg",
        "sample.jpg",
        "data/raw/test.jpg",
        "dataset/images/-10814178347722472.png",  # Your test image
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"Testing with {img_path}")
            return analyzer.analyze_and_print(img_path)
    
    print("No test image found. Please provide a path to test:")
    img_path = input("Enter image path: ").strip()
    if os.path.exists(img_path):
        return analyzer.analyze_and_print(img_path)
    else:
        print("File not found.")
        return None


if __name__ == "__main__":
    result = test_with_sample_image()
    if result:
        print("\n" + "="*60)
        print("Raw result (for API use):")
        print(result)