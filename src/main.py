import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

# Charger les labels depuis le fichier CSV
data = pd.read_csv("../dataset/cat_dog_shred.csv")
data["image"] = data["image"].apply(lambda x: os.path.join("../dataset/images", x))  # Ajuster le chemin si nécessaire

# Verifier le nombres d'image dans chaque set
print(f"Nombre d'images de chaque classe\n{data['labels'].value_counts()}")

######################
#      AlexNet       #
######################

# Définir les transformations pour AlexNet
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip images
    transforms.RandomRotation(10),     # Rotate images randomly
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Adjust brightness, contrast, etc.
    transforms.Resize((224, 224)),     # Resize images to 224x224
    transforms.ToTensor(),             # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
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

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(data['labels']),
    y=data['labels']
)

# Convert to tensor for PyTorch
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

import torch.optim as optim
from torch.nn import CrossEntropyLoss

# Fonction de perte et optimiseur
criterion = CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(alexnet.parameters(), lr=1e-4)

# Fonction d'entraînement
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Entraînement
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistiques
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)
        print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print("-" * 10)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_corrects.double() / len(dataloader.dataset)
    return total_loss, total_acc

# Dictionnaire des DataLoaders
dataloaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
}

# print("Training AlexNet")
#
# # Entraîner le modèle
# train_model(alexnet, dataloaders, criterion, optimizer, device, num_epochs=10)
#
# # Évaluer sur le test set
# test_loss, test_acc = evaluate_model(alexnet, dataloaders['test'], criterion, device)
# print(f"Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}")

######################
#      Custom        #
######################

from homemade_cnn import CustomCNN

# Initialiser le modèle
model = CustomCNN(num_classes=2)

# Déplacer sur GPU si disponible
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("Training Custom CNN")

# Entraîner le modèle
train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10)

# Évaluer sur le test set
test_loss, test_acc = evaluate_model(alexnet, dataloaders['test'], criterion, device)
print(f"Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}")
