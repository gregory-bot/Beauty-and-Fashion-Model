import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models.vision.train_v2 import ImprovedBeautyVisionModel
from models.vision.dataset import BeautyVisionDataset

def evaluate_v2_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = ImprovedBeautyVisionModel().to(device)
    model_path = "models/vision/beauty_vision_model_v2.pt"
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best validation accuracy - Skin: {checkpoint['val_skin_acc']:.4f}, Fitz: {checkpoint['val_fitz_acc']:.4f}")
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = BeautyVisionDataset(
        csv_path="data/processed/beauty_ml_vision_dataset.csv",
        transform=transform
    )
    
    # Create test set (last 20% of data)
    from sklearn.model_selection import train_test_split
    all_indices = list(range(len(dataset)))
    _, test_idx = train_test_split(all_indices, test_size=0.2, random_state=42, shuffle=True)
    
    from torch.utils.data import Subset
    test_dataset = Subset(dataset, test_idx)
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\nTest samples: {len(test_dataset)}")
    
    # Evaluate
    all_skin_preds = []
    all_skin_targets = []
    all_fitz_preds = []
    all_fitz_targets = []
    all_skin_probs = []
    all_fitz_probs = []
    
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            skin_targets = labels["monk_skin_tone"].cpu().numpy() - 1
            fitz_targets = labels["fitzpatrick"].cpu().numpy() - 1
            
            outputs = model(images)
            
            # Get predictions and probabilities
            skin_probs = F.softmax(outputs["skin_tone"], dim=1)
            fitz_probs = F.softmax(outputs["fitzpatrick"], dim=1)
            
            skin_preds = torch.argmax(skin_probs, dim=1).cpu().numpy()
            fitz_preds = torch.argmax(fitz_probs, dim=1).cpu().numpy()
            
            all_skin_preds.extend(skin_preds)
            all_skin_targets.extend(skin_targets)
            all_fitz_preds.extend(fitz_preds)
            all_fitz_targets.extend(fitz_targets)
            
            all_skin_probs.extend(skin_probs.cpu().numpy())
            all_fitz_probs.extend(fitz_probs.cpu().numpy())
    
    # Calculate metrics
    skin_accuracy = np.mean(np.array(all_skin_preds) == np.array(all_skin_targets))
    fitz_accuracy = np.mean(np.array(all_fitz_preds) == np.array(all_fitz_targets))
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    print(f"\nAccuracy Scores:")
    print(f"  Skin Tone (Monk 1-10): {skin_accuracy:.4f}")
    print(f"  Fitzpatrick (1-6): {fitz_accuracy:.4f}")
    
    print(f"\nImprovement over original model:")
    print(f"  Skin Tone: +{(skin_accuracy - 0.2884)*100:.1f}%")
    print(f"  Fitzpatrick: +{(fitz_accuracy - 0.3159)*100:.1f}%")
    
    # Classification reports
    print("\nSkin Tone Classification Report:")
    print(classification_report(
        all_skin_targets, 
        all_skin_preds,
        target_names=[f"Monk_{i+1}" for i in range(10)]
    ))
    
    print("\nFitzpatrick Classification Report:")
    print(classification_report(
        all_fitz_targets, 
        all_fitz_preds,
        target_names=[f"FST_{i+1}" for i in range(6)]
    ))
    
    # Save results
    results = {
        'skin_accuracy': float(skin_accuracy),
        'fitz_accuracy': float(fitz_accuracy),
        'skin_predictions': [int(x) for x in all_skin_preds],
        'skin_targets': [int(x) for x in all_skin_targets],
        'fitz_predictions': [int(x) for x in all_fitz_preds],
        'fitz_targets': [int(x) for x in all_fitz_targets],
    }
    
    import json
    results_path = "models/vision/test_results_v2.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return results

if __name__ == "__main__":
    evaluate_v2_model()