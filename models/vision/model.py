import torch
import torch.nn as nn
from torchvision import models


class BeautyVisionModel(nn.Module):
    def __init__(self, backbone_name='resnet18'):
        super().__init__()
        
        # Choose backbone
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
        elif backbone_name == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Remove original classifier
        self.backbone.fc = nn.Identity()
        
        # Heads
        self.skin_tone_head = nn.Linear(in_features, 10)      # Monk 1–10
        self.fitzpatrick_head = nn.Linear(in_features, 6)     # fst1–fst6

    def forward(self, x):
        features = self.backbone(x)

        skin_tone_logits = self.skin_tone_head(features)
        fitzpatrick_logits = self.fitzpatrick_head(features)

        return {
            "skin_tone": skin_tone_logits,
            "fitzpatrick": fitzpatrick_logits
        }


class ImprovedBeautyVisionModel(nn.Module):
    def __init__(self, backbone_name='resnet34'):
        super().__init__()
        
        # Load backbone
        if backbone_name == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Freeze only early layers (optional)
        for name, param in self.backbone.named_parameters():
            if 'layer1' in name or 'conv1' in name or 'bn1' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Remove original classifier
        self.backbone.fc = nn.Identity()
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Enhanced heads
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