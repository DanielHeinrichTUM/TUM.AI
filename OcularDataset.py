import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class OcularDataset(Dataset):
    def __init__(self, input_df, truth_df, transform=None):
        self.input_df = input_df
        self.truth_df = truth_df
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.input_df)
    
    def __getitem__(self, idx):
        # Get image paths (using right fundus as example)
        left_img_path = self.input_df.iloc[idx]["Left-Fundus"]
        right_img_path = self.input_df.iloc[idx]["Right-Fundus"]
        
        # Load images
        left_img = Image.open(left_img_path).convert('RGB')
        right_img = Image.open(right_img_path).convert('RGB')
        
        # Apply transforms
        left_img = self.transform(left_img)
        right_img = self.transform(right_img)
        
        # Stack images (assuming you want to use both eyes)
        images = torch.stack([left_img, right_img])
        
        # Get labels (convert to float32 for BCE loss)
        labels = torch.FloatTensor(
            self.truth_df.iloc[idx][["N","D","G","C","A","H","M","O"]].values
        )
        
        # Get additional metadata if needed
        sex = 0 if self.input_df.iloc[idx]["Patient Sex"] == "Male" else 1
        
        return {
            "images": images,          # Shape: [2, 3, 224, 224]
            "labels": labels,          # Shape: [8]
            "sex": torch.tensor(sex),  # 0/1
            "id": self.input_df.iloc[idx]["ID"]
        }