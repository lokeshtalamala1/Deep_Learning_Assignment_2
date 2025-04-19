import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.utils import shuffle
import os
import cv2
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import wandb


# Configuration
CLASSES = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", 
          "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]
IMAGE_SIZE = 500
NORMALIZATION_MEAN = 0.5
NORMALIZATION_STD = 0.5
ACTIVATIONS = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data loading
def create_data_loaders(train_dir, test_dir, batch_size, augment_data=False):
    """Create stratified train/val/test dataloaders with optional augmentation."""
    
    base_transforms = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((NORMALIZATION_MEAN,) * 3, (NORMALIZATION_STD,) * 3)
    ]
    
    train_transforms = base_transforms.copy()
    if augment_data:
        train_transforms = [
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomRotation(10),
        ] + train_transforms

    train_dataset = datasets.ImageFolder(train_dir, transform=transforms.Compose(train_transforms))
    test_dataset = datasets.ImageFolder(test_dir, transform=transforms.Compose(base_transforms))

    train_indices, val_indices = stratified_split(train_dataset)
    
    return (
        DataLoader(Subset(train_dataset, train_indices), batch_size, True, num_workers=4),
        DataLoader(Subset(train_dataset, val_indices), batch_size, False, num_workers=4),
        DataLoader(test_dataset, batch_size, False, num_workers=4)
    )

def stratified_split(dataset, val_size=0.2):
    """Create stratified indices using sklearn's train_test_split."""
    targets = [label for _, label in dataset.samples]
    return train_test_split(
        np.arange(len(targets)),
        test_size=val_size,
        stratify=targets,
        random_state=42
    )


# Model architecture
class BiodiversityCNN(nn.Module):
    """CNN for biological taxonomy classification."""
    
    def __init__(self, filter_config, kernel_sizes, activation, 
                 dense_units, dropout_prob, use_batchnorm):
        super().__init__()
        
        self.feature_extractor = self._build_conv_stack(
            filter_config, kernel_sizes, ACTIVATIONS[activation], use_batchnorm
        )
        self.classifier = nn.Sequential(
            nn.Linear(self._calculate_conv_output(filter_config), 
            nn.Dropout(dropout_prob),
            nn.Linear(dense_units, len(CLASSES))
    
    def _build_conv_stack(self, filters, kernels, activation, use_batchnorm):
        layers = []
        in_channels = 3
        for out_channels, kernel in zip(filters, kernels):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel),
                activation(),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
            ])
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def _calculate_conv_output(self, filters):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            return self.feature_extractor(dummy).view(1, -1).size(1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features.flatten(1))


# Training infrastructure
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs):
    """Training loop with validation and progress tracking."""
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        
        validate(model, criterion, train_loader, "train")
        validate(model, criterion, val_loader, "val")

def validate(model, criterion, loader, phase):
    """Validation/metrics reporting."""
    model.eval()
    total_loss, correct = 0.0, 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
    
    metrics = {
        f"{phase}_loss": total_loss / len(loader),
        f"{phase}_acc": 100 * correct / len(loader.dataset)
    }
    print(f"{phase.title()}: Loss={metrics[f'{phase}_loss']:.4f}, Acc={metrics[f'{phase}_acc']:.2f}%")
    wandb.log(metrics)


# Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Biodiversity Classification Trainer")
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=["adam", "nadam", "sgd"], default="nadam")
    parser.add_argument("--wandb_project", required=True)
    parser.add_argument("--wandb_entity", required=True)
    args = parser.parse_args()
    
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        "train", "test", args.batch_size, augment_data=True
    )
    
    model = BiodiversityCNN(
        filter_config=[128,128,64,64,32],
        kernel_sizes= [3,3,3,3,3],
        activation="elu",
        dense_units=512,
        dropout_prob=0.4,
        use_batchnorm=True
    )
    
    train_model(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.NAdam(model.parameters(), lr=args.lr),
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs
    )
    
    wandb.finish()
