import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.utils import shuffle # for shuffling
import os
import cv2
import random
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import gc
import argparse
import wandb


# Configuration Dataclass
class TrainingConfig:
    """Container for training hyperparameters and settings"""
    def __init__(self, **kwargs):
        self.image_size = 224
        self.normalization_mean = [0.485, 0.456, 0.406]
        self.normalization_std = [0.229, 0.224, 0.225]
        self.num_classes = 10
        self.class_names = [
            "Amphibia", "Animalia", "Arachnida", "Aves", "Fungi",
            "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"
        ]
        self.__dict__.update(kwargs)


# Model Architecture
class FineTunedResNet:
    """Wrapper for pretrained ResNet50 with custom classifier"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self) -> nn.Module:
        """Construct and configure pretrained model"""
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze convolutional base
        for param in base_model.parameters():
            param.requires_grad = False
            
        # Replace final classification layer
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, self.config.num_classes)
        
        return base_model

# Data Pipeline
class DataManager:
    """Handles dataset preparation and loading"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.transform = self._get_transforms()
        
    def _get_transforms(self) -> transforms.Compose:
        """Create image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalization_mean,
                std=self.config.normalization_std
            )
        ])
    
    def create_loaders(self, data_path: Path, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """Generate stratified data loaders"""
        full_dataset = datasets.ImageFolder(data_path, transform=self.transform)
        train_indices, val_indices = self._stratified_split(full_dataset)
        
        return (
            self._create_loader(Subset(full_dataset, train_indices), batch_size, shuffle=True),
            self._create_loader(Subset(full_dataset, val_indices), batch_size, shuffle=False)
        )
    
    @staticmethod
    def _stratified_split(dataset: datasets.ImageFolder, test_size: float = 0.2) -> Tuple[list, list]:
        """Create balanced train/validation splits"""
        class_indices = {}
        for idx, (_, label) in enumerate(dataset.samples):
            class_indices.setdefault(label, []).append(idx)
            
        return (
            [i for cls in class_indices.values() for i in train_test_split(cls, test_size=test_size)[0]],
            [i for cls in class_indices.values() for i in train_test_split(cls, test_size=test_size)[1]]
        )
    
    @staticmethod
    def _create_loader(dataset: Subset, batch_size: int, shuffle: bool) -> DataLoader:
        """Create optimized DataLoader instance"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )


# Training Engine
class TrainingEngine:
    """Orchestrates model training and evaluation"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def configure_optimizer(self, optimizer_name: str, lr: float, wd: float) -> optim.Optimizer:
        """Initialize optimization algorithm"""
        optimizer_registry = {
            'adam': optim.Adam,
            'nadam': optim.NAdam,
            'rmsprop': optim.RMSprop,
            'sgd': optim.SGD
        }
        return optimizer_registry[optimizer_name](
            self.model.parameters(),
            lr=lr,
            weight_decay=wd
        )
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        num_epochs: int
    ) -> Dict[str, Any]:
        """Execute full training workflow"""
        best_accuracy = 0.0
        training_report = {}
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation phase
            val_metrics = self.evaluate_model(val_loader, criterion)
            
            # Update tracking metrics
            training_report[epoch] = {
                'train_loss': epoch_loss / len(train_loader),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            }
            
            # Log metrics to WandB
            wandb.log({
                'epoch': epoch,
                **training_report[epoch]
            })
            
            # Save best model
            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
                torch.save(self.model.state_dict(), 'best_model.pth')
        
        return training_report
    
    def evaluate_model(self, data_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Assess model performance on given dataset"""
        self.model.eval()
        total_loss = 0.0
        correct_preds = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                total_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels).item()
        
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': 100 * correct_preds / len(data_loader.dataset)
        }


# Main Execution
def main(args: argparse.Namespace):
    """Orchestrate training pipeline"""
    # Initialize configuration
    config = TrainingConfig(
        base_dir=Path(args.base_dir),
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs
    )
    
    # Setup WandB
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    
    try:
        # Data preparation
        data_manager = DataManager(config)
        train_loader, val_loader = data_manager.create_loaders(
            config.base_dir / "train",
            config.batch_size
        )
        
        # Model initialization
        model_wrapper = FineTunedResNet(config)
        training_engine = TrainingEngine(model_wrapper.model, config)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = training_engine.configure_optimizer(
            config.optimizer,
            config.learning_rate,
            config.weight_decay
        )
        
        # Execute training
        training_report = training_engine.train_model(
            train_loader,
            val_loader,
            criterion,
            optimizer,
            config.epochs
        )
        
        # Final evaluation
        final_metrics = training_engine.evaluate_model(val_loader, criterion)
        print(f"\nFinal Validation Accuracy: {final_metrics['accuracy']:.2f}%")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Command line interface
    parser = argparse.ArgumentParser(description="Biodiversity Classification Trainer")
    
    parser.add_argument("--wandb_entity", "-we",
                        default="cs24m023-indian-institute-of-technology-madras",
                        help="Weights & Biases entity for experiment tracking")
    parser.add_argument("--wandb_project", "-wp",
                        default="Deep_Learning_Assignment_2",
                        help="Project name for experiment tracking")
    parser.add_argument("--epochs", "-e",
                        type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", "-b",
                        type=int, default=32,
                        help="Input batch size for training")
    parser.add_argument("--optimizer", "-o",
                        choices=["adam", "nadam", "rmsprop", "sgd"],
                        default="nadam",
                        help="Optimization algorithm selection")
    parser.add_argument("--learning_rate", "-lr",
                        type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", "-wd",
                        type=float, default=0.005,
                        help="L2 regularization strength")
    parser.add_argument("--base_dir", "-br",
                        type=Path, default="inaturalist_12K",
                        help="Root directory containing training data")
    
    args = parser.parse_args()
    
    # Validate data directory
    if not (args.base_dir / "train").exists():
        raise FileNotFoundError(f"Training directory not found in {args.base_dir}")
    
    # Execute training pipeline
    wandb.login()
    main(args)
    wandb.finish()