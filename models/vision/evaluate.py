import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms  # Add this import
from sklearn.metrics import classification_report
from models.vision.dataset import BeautyVisionDataset
from models.vision.model import BeautyVisionModel


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = BeautyVisionModel().to(device)
    model_path = "models/vision/beauty_vision_model.pt"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        print("Please train the model first using: python -m models.vision.train")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.eval()
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    try:
        dataset = BeautyVisionDataset(
            csv_path="data/processed/beauty_ml_vision_dataset.csv",
            transform=transform
        )
        print(f"Dataset loaded with {len(dataset)} samples")
    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )
    
    # Evaluate
    all_skin_preds = []
    all_skin_targets = []
    all_fitz_preds = []
    all_fitz_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            skin_targets = labels["monk_skin_tone"].cpu().numpy() - 1
            fitz_targets = labels["fitzpatrick"].cpu().numpy() - 1
            
            outputs = model(images)
            
            skin_preds = torch.argmax(outputs["skin_tone"], dim=1).cpu().numpy()
            fitz_preds = torch.argmax(outputs["fitzpatrick"], dim=1).cpu().numpy()
            
            all_skin_preds.extend(skin_preds)
            all_skin_targets.extend(skin_targets)
            all_fitz_preds.extend(fitz_preds)
            all_fitz_targets.extend(fitz_targets)
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}")
    
    # Print evaluation metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print("\nSkin Tone (Monk Scale 1-10) Classification Report:")
    print("Note: Classes are 0-9 in model output (mapped to 1-10 in Monk scale)")
    print(classification_report(
        all_skin_targets, 
        all_skin_preds,
        target_names=[f"Monk_{i+1}" for i in range(10)]
    ))
    
    print("\nFitzpatrick Scale (1-6) Classification Report:")
    print("Note: Classes are 0-5 in model output (mapped to 1-6 in Fitzpatrick scale)")
    print(classification_report(
        all_fitz_targets, 
        all_fitz_preds,
        target_names=[f"FST_{i+1}" for i in range(6)]
    ))
    
    # Calculate accuracy
    skin_accuracy = sum(1 for p, t in zip(all_skin_preds, all_skin_targets) if p == t) / len(all_skin_targets)
    fitz_accuracy = sum(1 for p, t in zip(all_fitz_preds, all_fitz_targets) if p == t) / len(all_fitz_targets)
    
    print(f"\nOverall Skin Tone Accuracy: {skin_accuracy:.4f}")
    print(f"Overall Fitzpatrick Accuracy: {fitz_accuracy:.4f}")
    
    # Show sample predictions
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS (first 10 samples):")
    print("="*50)
    for i in range(min(10, len(all_skin_preds))):
        print(f"Sample {i+1}: Skin Tone Pred={all_skin_preds[i]+1} (True={all_skin_targets[i]+1}) | "
              f"Fitzpatrick Pred={all_fitz_preds[i]+1} (True={all_fitz_targets[i]+1})")
    
    return {
        "skin_accuracy": skin_accuracy,
        "fitz_accuracy": fitz_accuracy,
        "skin_predictions": all_skin_preds,
        "skin_targets": all_skin_targets,
        "fitz_predictions": all_fitz_preds,
        "fitz_targets": all_fitz_targets
    }


if __name__ == "__main__":
    # Also need to import sklearn if not already installed
    try:
        from sklearn.metrics import classification_report
    except ImportError:
        print("scikit-learn not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
        from sklearn.metrics import classification_report
    
    evaluate_model()