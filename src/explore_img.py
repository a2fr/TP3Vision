import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

# Charger les labels depuis le fichier CSV
data = pd.read_csv("../dataset/cat_dog.csv")
data["image"] = data["image"].apply(lambda x: os.path.join("../dataset/images", x))  # Ajuster le chemin si nécessaire

# Définir les transformations pour AlexNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionnement
    transforms.ToTensor(),  # Convertir en tenseur
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation ImageNet
])

# Dataset personnalisé pour charger et transformer les images
class CatDogDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")  # Convertir en RGB
        if self.transform:
            image = self.transform(image)
        return image, label

from sklearn.model_selection import train_test_split

# Diviser les données en train (70%), temp (30%)
train_data, temp_data = train_test_split(data, test_size=0.3, stratify=data["labels"], random_state=42)

# Diviser temp (30%) en val (20%) et test (10%)
val_data, test_data = train_test_split(temp_data, test_size=0.33, stratify=temp_data["labels"], random_state=42)

# Créer les datasets PyTorch
train_dataset = CatDogDataset(train_data, transform=transform)
val_dataset = CatDogDataset(val_data, transform=transform)
test_dataset = CatDogDataset(test_data, transform=transform)

from torch.utils.data import DataLoader

# Définir les DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch
import torch.nn as nn
from torchvision import models

# Charger AlexNet pré-entraîné
alexnet = models.alexnet(pretrained=True)

# Adapter la dernière couche pour deux classes (chats et chiens)
alexnet.classifier[6] = nn.Linear(4096, 2)

# Déplacer le modèle sur GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alexnet = alexnet.to(device)
