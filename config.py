import os
import torch
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Model configuration
MODEL_CONFIG = {
    "num_classes": 6,  # Update based on your dataset
    "image_size": (800, 800),
    "confidence_threshold": 0.7,
    "iou_threshold": 0.5,
}

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 4,
    "num_epochs": 10,
    "learning_rate": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
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
