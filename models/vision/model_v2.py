import torch
import torch.nn as nn
from torchvision import models

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