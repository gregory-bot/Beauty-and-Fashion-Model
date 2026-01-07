import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import Counter

from models.vision.dataset import BeautyVisionDataset
from models.vision.model import BeautyVisionModel


# =========================
# CONFIG
# =========================
CSV_PATH = "data/processed/beauty_ml_vision_dataset.csv"
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "models/vision/beauty_vision_model.pt"


# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =========================
# DATASET & DATALOADER
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

dataset = BeautyVisionDataset(
    csv_path=CSV_PATH,
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# =========================
# CLASS WEIGHTS (FAIRNESS)
# =========================
def compute_class_weights(values, num_classes):
    counter = Counter(values)
    total = sum(counter.values())
    return torch.tensor(
        [total / counter.get(i, 1) for i in range(1, num_classes + 1)],
        dtype=torch.float
    )


skin_tone_weights = compute_class_weights(
    dataset.df["monk_skin_tone_label_us"].tolist(), 10
).to(device)

fitz_weights = compute_class_weights(
    dataset.df["fitzpatrick_label"].tolist(), 6
).to(device)


# =========================
# MODEL, LOSS, OPTIMIZER
# =========================
model = BeautyVisionModel().to(device)

criterion_skin = nn.CrossEntropyLoss(weight=skin_tone_weights)
criterion_fitz = nn.CrossEntropyLoss(weight=fitz_weights)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)


# =========================
# TRAINING LOOP
# =========================
model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        skin_targets = labels["monk_skin_tone"].to(device) - 1
        fitz_targets = labels["fitzpatrick"].to(device) - 1

        optimizer.zero_grad()
        outputs = model(images)

        loss = (
            criterion_skin(outputs["skin_tone"], skin_targets) +
            criterion_fitz(outputs["fitzpatrick"], fitz_targets)
        )

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss / len(dataloader):.4f}")


# =========================
# SAVE MODEL
# =========================
os.makedirs("models/vision", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
