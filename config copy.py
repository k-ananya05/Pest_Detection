import os
import torch
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Model configuration
MODEL_CONFIG = {
    "num_classes": 103,  # 102 classes (0-101) + 1 for background
    "image_size": (600, 600),  # Reduced from 800x800 for faster processing
    "confidence_threshold": 0.7,
    "iou_threshold": 0.5,
}

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 8,           # Increased batch size for faster training
    "num_epochs": 15,          # Slightly more epochs with faster training
    "learning_rate": 0.02,     # Increased learning rate
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,       # For learning rate warmup
    "num_workers": 4,         # For faster data loading
}

# Paths
PATHS = {
    "model_save": BASE_DIR / "saved_models",
    "results": BASE_DIR / "results",
    "data": BASE_DIR / "data",
}

# Create directories if they don't exist
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
