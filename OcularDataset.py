import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class OcularDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        if df is None:
            raise ValueError("DataFrame (df) cannot be None!")
        self.df = df
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        self.sex_mapping = {"Male": 0, "Female": 1}
        self.age_mean, self.age_std = df["Patient Age"].mean(), df["Patient Age"].std()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        left_img = self._load_image(row["Left-Fundus"])
        right_img = self._load_image(row["Right-Fundus"])
        
        labels = row[["N","D","G","C","A","H","M","O"]].values.astype(float)
        sex = self.sex_mapping.get(row["Patient Sex"], 0) #defaulting to male if unknown, just like in real medical research
        age = (row["Patient Age"] - self.age_mean) / self.age_std
        
        return {
            "images": torch.stack([left_img, right_img]),
            "labels": torch.FloatTensor(labels),
            "sex_age": torch.FloatTensor([sex, age]),
            "id": row["ID"]
        }
    
    def _load_image(self, filename):
        img_path = os.path.join(self.image_dir, filename)
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)