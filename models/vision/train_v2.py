import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from collections import Counter
import numpy as np

# Add project root to Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.vision.dataset import BeautyVisionDataset

# =========================
# IMPROVED CONFIG
# =========================
CSV_PATH = "data/processed/beauty_ml_vision_dataset.csv"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "models/vision/beauty_vision_model_v2.pt"

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# IMPROVED TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================
# DATASET & DATALOADER
# =========================
print("Loading dataset...")
train_dataset = BeautyVisionDataset(
    csv_path=CSV_PATH,
    transform=train_transform
)

val_dataset = BeautyVisionDataset(
    csv_path=CSV_PATH,
    transform=val_transform
)

all_indices = list(range(len(train_dataset)))

from sklearn.model_selection import train_test_split
train_idx, val_idx = train_test_split(all_indices, test_size=0.2, random_state=42)

from torch.utils.data import Subset
train_subset = Subset(train_dataset, train_idx)
val_subset = Subset(val_dataset, val_idx)

train_dataloader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_dataloader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Training samples: {len(train_subset)}")
print(f"Validation samples: {len(val_subset)}")

# =========================
# IMPROVED MODEL
# =========================
class ImprovedBeautyVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Freeze only early layers
        for name, param in self.backbone.named_parameters():
            if 'layer1' in name or 'conv1' in name or 'bn1' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.dropout = nn.Dropout(0.5)
        
        self.skin_tone_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
        
        self.fitzpatrick_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        
        skin_tone_logits = self.skin_tone_head(features)
        fitzpatrick_logits = self.fitzpatrick_head(features)
        
        return {
            "skin_tone": skin_tone_logits,
            "fitzpatrick": fitzpatrick_logits
        }

# =========================
# CLASS WEIGHTS
# =========================
def compute_class_weights_with_smoothing(values, num_classes, smoothing=10.0):
    counter = Counter(values)
    total = sum(counter.values())
    
    weights = []
    for i in range(1, num_classes + 1):
        count = counter.get(i, 0)
        weight = total / (count + smoothing)
        weights.append(weight)
    
    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.sum() * num_classes
    return weights

train_labels_monk = [train_dataset.df.iloc[idx]["monk_skin_tone_label_us"] for idx in train_idx]
train_labels_fitz = [train_dataset.df.iloc[idx]["fitzpatrick_label"] for idx in train_idx]

skin_tone_weights = compute_class_weights_with_smoothing(train_labels_monk, 10).to(device)
fitz_weights = compute_class_weights_with_smoothing(train_labels_fitz, 6).to(device)

print(f"Class weights - Skin Tone: {skin_tone_weights}")
print(f"Class weights - Fitzpatrick: {fitz_weights}")

# =========================
# MODEL, LOSS, OPTIMIZER
# =========================
model = ImprovedBeautyVisionModel().to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

criterion_skin = nn.CrossEntropyLoss(weight=skin_tone_weights)
criterion_fitz = nn.CrossEntropyLoss(weight=fitz_weights)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# FIXED: Remove verbose=True
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# =========================
# TRAINING LOOP
# =========================
best_val_loss = float('inf')
best_val_acc = 0.0

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    train_skin_correct = 0
    train_fitz_correct = 0
    train_total = 0
    
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        skin_targets = labels["monk_skin_tone"].to(device) - 1
        fitz_targets = labels["fitzpatrick"].to(device) - 1
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss_skin = criterion_skin(outputs["skin_tone"], skin_targets)
        loss_fitz = criterion_fitz(outputs["fitzpatrick"], fitz_targets)
        loss = loss_skin + loss_fitz
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
        
        _, skin_preds = torch.max(outputs["skin_tone"], 1)
        _, fitz_preds = torch.max(outputs["fitzpatrick"], 1)
        
        train_skin_correct += (skin_preds == skin_targets).sum().item()
        train_fitz_correct += (fitz_preds == fitz_targets).sum().item()
        train_total += skin_targets.size(0)
        
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(train_dataloader)} - Loss: {loss.item():.4f}")
    
    avg_train_loss = train_loss / len(train_dataloader)
    train_skin_acc = train_skin_correct / train_total
    train_fitz_acc = train_fitz_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_skin_correct = 0
    val_fitz_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            skin_targets = labels["monk_skin_tone"].to(device) - 1
            fitz_targets = labels["fitzpatrick"].to(device) - 1
            
            outputs = model(images)
            
            loss_skin = criterion_skin(outputs["skin_tone"], skin_targets)
            loss_fitz = criterion_fitz(outputs["fitzpatrick"], fitz_targets)
            loss = loss_skin + loss_fitz
            val_loss += loss.item()
            
            _, skin_preds = torch.max(outputs["skin_tone"], 1)
            _, fitz_preds = torch.max(outputs["fitzpatrick"], 1)
            
            val_skin_correct += (skin_preds == skin_targets).sum().item()
            val_fitz_correct += (fitz_preds == fitz_targets).sum().item()
            val_total += skin_targets.size(0)
    
    avg_val_loss = val_loss / len(val_dataloader)
    val_skin_acc = val_skin_correct / val_total
    val_fitz_acc = val_fitz_correct / val_total
    
    # Update scheduler
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] - LR: {current_lr:.2e}")
    print(f"  Train Loss: {avg_train_loss:.4f} | Skin Acc: {train_skin_acc:.4f} | Fitz Acc: {train_fitz_acc:.4f}")
    print(f"  Val Loss:   {avg_val_loss:.4f} | Skin Acc: {val_skin_acc:.4f} | Fitz Acc: {val_fitz_acc:.4f}")
    
    # Save best model
    if val_skin_acc + val_fitz_acc > best_val_acc:
        best_val_acc = val_skin_acc + val_fitz_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_skin_acc': val_skin_acc,
            'val_fitz_acc': val_fitz_acc,
            'val_loss': avg_val_loss,
        }, MODEL_SAVE_PATH)
        print(f"  💾 BEST MODEL SAVED to {MODEL_SAVE_PATH}")
        print(f"     Skin Acc: {val_skin_acc:.4f}, Fitz Acc: {val_fitz_acc:.4f}")

print("\n" + "="*60)
print(f"TRAINING COMPLETED")
print(f"Best model saved to {MODEL_SAVE_PATH}")
print("="*60)

# =========================
# FINAL EVALUATION
# =========================
print("\n" + "="*60)
print("FINAL EVALUATION ON VALIDATION SET")
print("="*60)

# Load best model
checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
all_skin_preds = []
all_skin_targets = []
all_fitz_preds = []
all_fitz_targets = []

with torch.no_grad():
    for images, labels in val_dataloader:
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

skin_accuracy = np.mean(np.array(all_skin_preds) == np.array(all_skin_targets))
fitz_accuracy = np.mean(np.array(all_fitz_preds) == np.array(all_fitz_targets))

print(f"\nFinal Validation Accuracy:")
print(f"  Skin Tone (Monk 1-10): {skin_accuracy:.4f}")
print(f"  Fitzpatrick (1-6): {fitz_accuracy:.4f}")

print("\nClass Distribution in Validation Set:")
print(f"  Skin Tone: {Counter([x+1 for x in all_skin_targets])}")
print(f"  Fitzpatrick: {Counter([x+1 for x in all_fitz_targets])}")

# Save final metrics
metrics = {
    'skin_accuracy': float(skin_accuracy),
    'fitz_accuracy': float(fitz_accuracy),
    'best_val_acc': float(best_val_acc),
    'best_epoch': checkpoint['epoch']
}

import json
metrics_path = MODEL_SAVE_PATH.replace('.pt', '_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to {metrics_path}")