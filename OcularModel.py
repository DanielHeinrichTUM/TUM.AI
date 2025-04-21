"""Setup of the model, resnet50 with only the last layer changed to a fc classification layer"""

import torch
import torchvision
from torch import nn

class OcularModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Image feature extractor (ResNet50)
        self.backbone = torchvision.models.resnet50(weights='DEFAULT')
        self.backbone.fc = nn.Identity()  # Remove original FC layer
        
        # Demographic branch (sex + age)
        self.demographic_fc = nn.Sequential(
            nn.Linear(2, 32),  # Input: [sex (0/1), age (normalized)]
            nn.ReLU()
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048*2 + 32, 512),  # 2048*2 (both eyes) + 32 (demographics)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 8)  # 8 diseases (output logits)
        )

    def forward(self, x_images, x_sex_age):
        # Extract image features (both eyes)
        left = self.backbone(x_images[:, 0])  # [batch, 2048]
        right = self.backbone(x_images[:, 1])  # [batch, 2048]
        
        # Process demographics
        demo_features = self.demographic_fc(x_sex_age)  # [batch, 32]
        
        # Combine and classify
        combined = torch.cat([left, right, demo_features], dim=1)  # [batch, 2048*2 + 32]
        return self.classifier(combined)  # [batch, 8]
