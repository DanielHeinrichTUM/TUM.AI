"""Setup of the model, resnet50 with only the last layer changed to a fc classification layer"""

import torch
import torchvision
from torch import nn

class OcularModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(weights='DEFAULT')
        
        for param in self.model.parameters():
            param.requires_grad = False
        num_classes = 8
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(2048*2, 512),  # 2048 features per eye * 2
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 8)  # 8 disease classes
        )
        
    def forward(self, x):
        """Input shape: [batch_size, 2, 3, 224, 224]"""
        left_features = self.backbone(x[:, 0])  # [batch, 2048]
        right_features = self.backbone(x[:, 1])  # [batch, 2048]
        combined = torch.cat([left_features, right_features], dim=1)
        return self.classifier(combined)
    


