import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('OcularDataset')

class OcularDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, default_img_size=(224, 224)):
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
        self.default_img_size = default_img_size
        self._validate_images()
    
    def _validate_images(self):
        missing_images = 0
        total_images = 0
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            for eye in ["Left-Fundus", "Right-Fundus"]:
                if eye in row and isinstance(row[eye], str) and row[eye].strip():
                    total_images += 1
                    img_path = os.path.join(self.image_dir, row[eye])
                    if not os.path.isfile(img_path):
                        missing_images += 1
        
        if total_images > 0:
            logger.info(f"Found {missing_images} missing images out of {total_images} total images")
            self.missing_image_rate = missing_images / total_images
        else:
            logger.warning("No image paths found in the dataset")
            self.missing_image_rate = 0
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        left_img = self._load_image_safely(row["Left-Fundus"])
        right_img = self._load_image_safely(row["Right-Fundus"])
        
        labels = row[["N","D","G","C","A","H","M","O"]].values.astype(float)
        sex = self.sex_mapping.get(row["Patient Sex"], 0)  # defaulting to male if unknown
        age = (row["Patient Age"] - self.age_mean) / self.age_std
        
        return {
            "images": torch.stack([left_img, right_img]),
            "labels": torch.FloatTensor(labels),
            "sex_age": torch.FloatTensor([sex, age]),
            "id": row["ID"]
        }
    
    def _load_image_safely(self, filename):
        if not isinstance(filename, str) or not filename.strip():
            return self._create_placeholder_tensor()
            
        img_path = os.path.join(self.image_dir, filename)
        
        try:
            if os.path.isfile(img_path):
                img = Image.open(img_path).convert('RGB')
                return self.transform(img)
            else:
                logger.debug(f"Image file not found: {img_path}")
                return self._create_placeholder_tensor()
        except Exception as e:
            logger.debug(f"Error loading image {img_path}: {str(e)}")
            return self._create_placeholder_tensor()
        
    def _create_placeholder_tensor(self):
        """Create a black image tensor placeholder for missing images"""
        black_tensor = torch.zeros(3, self.default_img_size[0], self.default_img_size[1])
        # Apply normalization to match other images
        for c, (m, s) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
            black_tensor[c] = (0 - m) / s
        return black_tensor