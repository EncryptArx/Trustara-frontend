#!/usr/bin/env python3
"""
Advanced Anti-Overfitting Training Script
Solves overfitting issues and creates a robust deepfake detection model

FIXES applied:
- Adaptively load pretrained weights even when some linear layers differ in size
  (handles attention shape mismatches like 128->1280 vs 256->1280).
- Platform-aware DataLoader worker selection and robust fallback for Windows
  / DataLoader timeouts (avoids training interruption).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from pathlib import Path
import json
from PIL import Image
import random
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
import time
from collections import defaultdict
import multiprocessing
warnings.filterwarnings('ignore')
from tqdm import tqdm
import torch.cuda


# ---------- Helper functions for robust pretrained loading ----------

def adapt_tensor(pretrained_tensor: torch.Tensor, target_shape: torch.Size):
    """
    Copy overlapping portion of pretrained_tensor into new tensor with target_shape.
    Remaining elements are initialized with a suitable initializer (kaiming_uniform for weights,
    zeros for biases if 1D).
    """
    new_tensor = torch.empty(target_shape, dtype=pretrained_tensor.dtype)
    # Choose initializer based on dimension (linear weight vs bias)
    if new_tensor.dim() == 2:
        # Linear weight: (out, in)
        # Copy overlapping block
        min_rows = min(new_tensor.size(0), pretrained_tensor.size(0))
        min_cols = min(new_tensor.size(1), pretrained_tensor.size(1))
        # Initialize then copy
        nn.init.kaiming_uniform_(new_tensor, a=np.sqrt(5))
        new_tensor[:min_rows, :min_cols] = pretrained_tensor[:min_rows, :min_cols]
    elif new_tensor.dim() == 1:
        # bias
        new_tensor.zero_()
        min_len = min(new_tensor.size(0), pretrained_tensor.size(0))
        new_tensor[:min_len] = pretrained_tensor[:min_len]
    else:
        # fallback: try copying flattened overlap
        new_tensor_flat = new_tensor.view(-1)
        pt_flat = pretrained_tensor.view(-1)
        min_len = min(new_tensor_flat.size(0), pt_flat.size(0))
        new_tensor_flat.zero_()
        new_tensor_flat[:min_len] = pt_flat[:min_len]
        new_tensor = new_tensor_flat.view(target_shape)
    return new_tensor

def adapt_and_load_state_dict(model: nn.Module, pretrained_state: dict, device='cpu'):
    """
    Try to adapt mismatched tensors from pretrained_state to model's expected shapes.
    This will:
      - For each key present in both dicts: if shapes match, keep as-is.
      - If shapes differ and both are tensors (common with Linear layers), adapt by copying
        overlapping portions and initializing rest.
      - Then load the resulting state_dict with strict=False.
    Returns a tuple (loaded_keys, adapted_keys, skipped_keys)
    """
    model_state = model.state_dict()
    adapted = []
    loaded_keys = []
    skipped = []

    patched_state = {}
    for k, v in pretrained_state.items():
        if k not in model_state:
            # key doesn't exist in target model
            skipped.append(k)
            continue
        target = model_state[k]
        if isinstance(v, torch.Tensor) and isinstance(target, torch.Tensor):
            if v.shape == target.shape:
                patched_state[k] = v.to(device)
                loaded_keys.append(k)
            else:
                # Attempt to adapt for linear-like shapes and 1D biases
                try:
                    adapted_tensor = adapt_tensor(v.cpu(), target.shape)
                    patched_state[k] = adapted_tensor.to(device)
                    adapted.append(k)
                except Exception:
                    skipped.append(k)
        else:
            # Non-tensor or incompatible type: skip
            skipped.append(k)

    # For any model keys not filled by patched_state, keep model's original
    for k in model_state:
        if k not in patched_state:
            patched_state[k] = model_state[k]

    # Load patched dict
    try:
        model.load_state_dict(patched_state, strict=False)
    except Exception as e:
        # As a final fallback, attempt non-adapted load with strict=False
        try:
            model.load_state_dict(pretrained_state, strict=False)
        except Exception as e2:
            print(f"Warning: Unable to load pretrained state even with fallback: {e2}")
    return loaded_keys, adapted, skipped

# ---------- Model and utilities (unchanged structure, small safety tweaks) ----------

class AdvancedDeepFakeModel(nn.Module):
    """Advanced model with better regularization and architecture."""
    
    def __init__(self, num_classes=1, dropout_rate=0.5, attention_dim=256):
        super(AdvancedDeepFakeModel, self).__init__()
        
        # Load pretrained backbone
        # Note: using a fixed torchvision hub version to reduce breakage
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        
        # Freeze more layers to prevent overfitting
        for param in list(self.backbone.parameters())[:-25]:  # Freeze more layers
            param.requires_grad = False
        
        # Save extracted feature dimension (mobilenet_v2 default is 1280)
        self.feature_dim = 1280
        
        # Make attention dimension configurable to help when loading pretrained attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, attention_dim),
            nn.BatchNorm1d(attention_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(attention_dim, self.feature_dim),
            nn.Sigmoid()
        )
        
        # Enhanced classifier with more regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights properly."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # initialize attention linears if present
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone.features(x)
        att_weights = self.attention(features)
        # att_weights shape [batch, feature_dim] -> reshape to multiply spatial features
        features = features * att_weights.unsqueeze(-1).unsqueeze(-1)
        x = nn.functional.adaptive_avg_pool2d(features, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ImprovedFocalLoss(nn.Module):
    """Improved Focal Loss with better parameters for deepfake detection."""
    
    def __init__(self, alpha=0.5, gamma=3.0, reduction='mean'):
        super(ImprovedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdvancedAugmentation:
    """Advanced data augmentation to prevent overfitting."""
    
    def __init__(self, is_training=True):
        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __call__(self, image):
        return self.transform(image)

class RobustDataset(Dataset):
    """Robust dataset with comprehensive error handling."""
    
    def __init__(self, real_dir, fake_dir, transform=None, max_samples=None, balance_classes=True):
        self.transform = transform
        self.samples = []
        self.failed_samples = 0
        
        # Load real images
        print("Loading real images...")
        real_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            real_paths.extend(list(Path(real_dir).glob(ext)))
        
        if max_samples and balance_classes:
            real_paths = real_paths[:max_samples//2]
        
        # Load fake images
        print("Loading fake images...")
        fake_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            fake_paths.extend(list(Path(fake_dir).glob(ext)))
        
        if max_samples and balance_classes:
            fake_paths = fake_paths[:max_samples//2]
        
        # Add samples
        for path in real_paths:
            self.samples.append((str(path), 0))  # 0 for real
        
        for path in fake_paths:
            self.samples.append((str(path), 1))  # 1 for fake
        
        print(f"Dataset loaded: {len(self.samples)} samples")
        print(f"  Real: {len(real_paths)}")
        print(f"  Fake: {len(fake_paths)}")
        
        # Shuffle samples
        random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        try:
            # Load and validate image
            img = Image.open(path).convert("RGB")
            img.verify()  # Verify image integrity
            # Reopen because verify() can close the file
            img = Image.open(path).convert("RGB")
            
            # Check image dimensions
            if img.size[0] < 32 or img.size[1] < 32:
                raise ValueError("Image too small")
            
            # Apply transforms
            if self.transform:
                img = self.transform(img)
            
            return img, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            self.failed_samples += 1
            if self.failed_samples <= 10:  # Only print first 10 errors
                print(f"Error loading {path}: {e}")
            
            # Return dummy tensor for failed images
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, torch.tensor(label, dtype=torch.float32)

class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def calculate_metrics(predictions, targets):
    """Calculate comprehensive metrics."""
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Basic metrics
    accuracy = (predictions == targets).mean() if len(predictions) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(targets, predictions) if len(predictions) > 0 else 0.0
    
    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions) if len(predictions) > 0 else np.zeros((2,2), dtype=int)
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }

# ---------- DataLoader / platform helpers ----------

def get_optimal_num_workers(preferred=4):
    """
    Choose num_workers based on platform and CPU count.
    - On Windows, default to 0 to avoid multiprocessing issues.
    - On other platforms, use min(preferred, max(1, cpu_count-1))
    """
    try:
        cpus = multiprocessing.cpu_count()
    except Exception:
        cpus = 1
    if os.name == 'nt':
        # Windows: safe default is 0 or low number to avoid DataLoader worker timeouts
        return 0
    else:
        return min(preferred, max(1, cpus - 1))

def make_dataloader(dataset, batch_size, shuffle, device, num_workers=None, timeout=60, persistent_workers=True, prefetch_factor=2):
    """
    Create DataLoader with robust fallbacks:
    - If num_workers is None, choose platform-appropriate default.
    - If creating the DataLoader causes issues, fallback to num_workers=0 safe mode.
    """
    if num_workers is None:
        num_workers = get_optimal_num_workers(preferred=4)

    # If num_workers=0, timeout must be 0
    if num_workers == 0:
        timeout = 0

    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'timeout': timeout,
        'pin_memory': True if device.type == 'cuda' else False
    }

    if num_workers > 0:
        kwargs.update({
            'persistent_workers': persistent_workers,
            'prefetch_factor': prefetch_factor
        })

    try:
        loader = DataLoader(dataset, **kwargs)
        # quick sanity check
        it = iter(loader)
        _ = next(it)
        return loader
    except Exception as e:
        print(f"Warning: DataLoader creation/iteration failed with num_workers={num_workers}: {e}")
        print("Falling back to num_workers=0 and disabling persistent_workers/prefetch.")
        kwargs['num_workers'] = 0
        kwargs['timeout'] = 0
        kwargs.pop('persistent_workers', None)
        kwargs.pop('prefetch_factor', None)
        try:
            loader = DataLoader(dataset, **kwargs)
            return loader
        except Exception as e2:
            print(f"Critical: Fallback DataLoader also failed: {e2}")
            raise

# ---------- Training routine (main) ----------

def train_advanced_model():
    """Train advanced model with all anti-overfitting techniques."""
    
    print("=" * 80)
    print("ADVANCED ANTI-OVERFITTING TRAINING")
    print("=" * 80)
    print("Features:")
    print("• Uses existing pretrained model as starting point")
    print("• Advanced data augmentation")
    print("• Early stopping with patience")
    print("• Improved regularization")
    print("• Better loss function")
    print("• Comprehensive monitoring")
    print("=" * 80)
    
    # Configuration optimized for large dataset (193k+ images)
    config = {
        'batch_size': 32,  # Larger batch size for stability with large dataset
        'learning_rate': 3e-5,  # Lower learning rate for large dataset
        'weight_decay': 1e-4,  # Weight decay
        'dropout_rate': 0.5,  # Higher dropout
        'max_epochs': 20,  # Fewer epochs since we have more data
        'patience': 4,  # Early stopping patience
        'min_delta': 0.001,  # Minimum improvement
        'label_smoothing': 0.1,  # Label smoothing
        'gradient_clip': 1.0,  # Gradient clipping
        'max_samples': None,  # Use all available data (193k+ images)
        'dataloader_timeout': 120  # seconds; increase to avoid spurious timeouts
    }
    
    print(f"Configuration: {config}")
    print("=" * 80)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths
    real_dir = "data/images/real"
    fake_dir = "data/images/deepfake"
    
    # Check if pretrained model exists
    pretrained_path = "models/simple_deepfake_detector.pt.best"
    use_pretrained = Path(pretrained_path).exists()
    
    if use_pretrained:
        print(f"Found pretrained model: {pretrained_path}")
        print("Will use as starting point for training")
    else:
        print("No pretrained model found, starting from scratch")
    
    # Create datasets
    print("Loading datasets...")
    print(f"Real directory: {real_dir}")
    print(f"Fake directory: {fake_dir}")
    train_dataset = RobustDataset(
        real_dir, fake_dir, 
        transform=AdvancedAugmentation(is_training=True),
        max_samples=config['max_samples'],  # None = use all data
        balance_classes=True
    )
    
    # Split dataset
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Update validation dataset transform
    val_dataset.dataset.transform = AdvancedAugmentation(is_training=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Determine number of workers (platform-aware)
    num_workers = get_optimal_num_workers(preferred=4)
    print(f"Chosen num_workers for DataLoader: {num_workers} (platform: {os.name})")
    
    # Create data loaders with robust fallback
    train_loader = make_dataloader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        device=device,
        num_workers=num_workers,
        timeout=config['dataloader_timeout'],
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = make_dataloader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        device=device,
        num_workers=num_workers,
        timeout=config['dataloader_timeout'],
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Model setup
    # Make attention_dim configurable - default 256 (you can change if loading a pretrained
    # checkpoint requires a different attention dimension).
    model = AdvancedDeepFakeModel(dropout_rate=config['dropout_rate'], attention_dim=256).to(device)
    
    # Load pretrained weights if available
    if use_pretrained:
        try:
            print("Loading pretrained weights...")
            pretrained_state = torch.load(pretrained_path, map_location=device)
            # If pretrained_state is a dict with metadata, try to extract state_dict
            if isinstance(pretrained_state, dict) and 'state_dict' in pretrained_state:
                pretrained_state = pretrained_state['state_dict']
            # Adapt mismatched shapes and load
            loaded_keys, adapted_keys, skipped_keys = adapt_and_load_state_dict(model, pretrained_state, device=device)
            print(f"Pretrained load summary: loaded={len(loaded_keys)}, adapted={len(adapted_keys)}, skipped={len(skipped_keys)}")
            if adapted_keys:
                print(f"Adapted keys (shapes changed): {adapted_keys[:10]}{'...' if len(adapted_keys)>10 else ''}")
            print("Pretrained weights applied (with adaptation where necessary).")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights via adaptation: {e}")
            print("Attempting fallback load with strict=False...")
            try:
                model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
                print("Fallback pretrained load succeeded.")
            except Exception as e2:
                print(f"Warning: Fallback load also failed: {e2}")
                print("Starting from scratch...")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # Restart every 5 epochs
        T_mult=2,
        eta_min=1e-6
    )
    
    # Loss function
    criterion = ImprovedFocalLoss(alpha=0.5, gamma=3.0)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['patience'],
        min_delta=config['min_delta']
    )
    
    # Training variables
    best_val_balanced_acc = 0.0
    best_epoch = 0
    training_history = []
    
    print("Starting advanced training...")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(config['max_epochs']):
        epoch_start = time.time()
        
         # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']} [Train]", ncols=100)
        for batch_idx, (data, target) in enumerate(train_iter):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    output = model(data)
                    loss = criterion(output.squeeze(), target)
            else:
                output = model(data)
                loss = criterion(output.squeeze(), target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
            optimizer.step()

            train_loss += loss.item()

            # Collect predictions for metrics
            with torch.no_grad():
                pred = (torch.sigmoid(output.squeeze()) > 0.5).float()
                train_predictions.extend(pred.cpu().numpy())
                train_targets.extend(target.cpu().numpy())

            # Update tqdm bar
            avg_loss = train_loss / (batch_idx + 1)
            mem = f"{torch.cuda.memory_reserved() / 1e9:.2f} GB" if device.type == 'cuda' else "CPU"
            train_iter.set_postfix({"loss": f"{avg_loss:.4f}", "mem": mem})
        
               # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []

        val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']} [Val]", ncols=100)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_iter):
                data, target = data.to(device), target.to(device)

                if device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda'):
                        output = model(data)
                        loss = criterion(output.squeeze(), target)
                else:
                    output = model(data)
                    loss = criterion(output.squeeze(), target)

                val_loss += loss.item()
                pred = (torch.sigmoid(output.squeeze()) > 0.5).float()
                val_predictions.extend(pred.cpu().numpy())
                val_targets.extend(target.cpu().numpy())

                avg_loss = val_loss / (batch_idx + 1)
                val_iter.set_postfix({"loss": f"{avg_loss:.4f}"})

        
        # Calculate metrics
        train_metrics = calculate_metrics(train_predictions, train_targets)
        val_metrics = calculate_metrics(val_predictions, val_targets)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{config['max_epochs']} ({epoch_time:.1f}s)")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_metrics['accuracy']:.3f} | Train Balanced: {train_metrics['balanced_accuracy']:.3f}")
        print(f"Val Loss:   {val_loss/len(val_loader):.4f} | Val Acc:   {val_metrics['accuracy']:.3f} | Val Balanced:   {val_metrics['balanced_accuracy']:.3f}")
        # Safe indexing for recall in case of zero-division or single-class edgecases
        try:
            train_real_recall = train_metrics['recall'][0]
            train_fake_recall = train_metrics['recall'][1]
            val_real_recall = val_metrics['recall'][0]
            val_fake_recall = val_metrics['recall'][1]
        except Exception:
            train_real_recall = train_fake_recall = val_real_recall = val_fake_recall = 0.0
        print(f"Real - Train: {train_real_recall:.3f} | Val: {val_real_recall:.3f}")
        print(f"Fake - Train: {train_fake_recall:.3f} | Val: {val_fake_recall:.3f}")
        print(f"LR: {current_lr:.2e}")
        
        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_metrics['accuracy'],
            'train_balanced_acc': train_metrics['balanced_accuracy'],
            'val_loss': val_loss / len(val_loader),
            'val_acc': val_metrics['accuracy'],
            'val_balanced_acc': val_metrics['balanced_accuracy'],
            'train_real_recall': train_real_recall,
            'train_fake_recall': train_fake_recall,
            'val_real_recall': val_real_recall,
            'val_fake_recall': val_fake_recall,
            'lr': current_lr,
            'epoch_time': epoch_time
        })
        
        # Early stopping check & best model saving
        if val_metrics['balanced_accuracy'] > best_val_balanced_acc:
            best_val_balanced_acc = val_metrics['balanced_accuracy']
            best_epoch = epoch + 1
            
            # Save best model
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/advanced_deepfake_detector_best.pt')
            print(f"Saved best model! (Epoch {epoch+1}, Val Balanced Acc: {val_metrics['balanced_accuracy']:.3f})")
        
        # Check for early stopping
        if early_stopping(val_metrics['balanced_accuracy'], model):
            print(f"Early stopping triggered! Best epoch: {best_epoch}")
            break
        
        print("-" * 60)
    
    # Training completed
    total_time = time.time() - start_time
    
    print("=" * 80)
    print("TRAINING COMPLETED!")
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Best validation balanced accuracy: {best_val_balanced_acc:.3f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Model saved as: models/advanced_deepfake_detector_best.pt")
    
    # Save training history
    os.makedirs('models', exist_ok=True)
    with open('models/advanced_training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("Training history saved to: models/advanced_training_history.json")
    
    # Final evaluation
    print("\nFINAL EVALUATION:")
    print("-" * 40)
    if best_epoch > 0 and best_epoch <= len(training_history):
        final_metrics = training_history[best_epoch - 1]
        print(f"Best Validation Accuracy: {final_metrics['val_acc']:.3f}")
        print(f"Best Validation Balanced Accuracy: {final_metrics['val_balanced_acc']:.3f}")
        print(f"Real Image Recall: {final_metrics['val_real_recall']:.3f}")
        print(f"Fake Image Recall: {final_metrics['val_fake_recall']:.3f}")
    else:
        print("No recorded best epoch metrics.")
    
    print("=" * 80)
    print("SUCCESS! Advanced model training completed.")
    print("The new model should have significantly better generalization!")
    print("=" * 80)
    
    return best_val_balanced_acc, best_epoch

if __name__ == "__main__":
    try:
        best_acc, best_epoch = train_advanced_model()
        print(f"\n🎉 SUCCESS! Best model achieved {best_acc:.3f} validation accuracy at epoch {best_epoch}")
        print("The new model is saved and ready to use!")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
