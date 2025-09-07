import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import cv2

class IP102Dataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, img_size=800):
        """
        IP102 Dataset in YOLO format.
        
        Args:
            data_dir: Path to the dataset root directory (should contain 'images' and 'labels' folders)
            split: 'train' or 'val'
            transform: Optional transform to be applied on a sample
            img_size: Target image size (assumes square images)
        """
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        
        # Set up paths
        self.img_dir = os.path.join(data_dir, 'images', split)
        self.label_dir = os.path.join(data_dir, 'labels', split)
        
        # Get list of image files
        self.img_files = [f for f in os.listdir(self.img_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Base transforms - always end with ToTensor()
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        # Data augmentation for training
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),  # Ensure we always end with a tensor
        ])
        
        # Load class names from YAML
        yaml_path = os.path.join(data_dir, 'ip102.yaml')
        with open(yaml_path, 'r') as f:
            import yaml
            data = yaml.safe_load(f)
            self.classes = data['names']
            self.num_classes = len(self.classes)
            
        # Debug: Print transform info
        print(f"Dataset initialized with {len(self.img_files)} {split} samples")
        print(f"Using {'training' if split == 'train' else 'validation'} transforms")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Open and convert to RGB first
        image = Image.open(img_path).convert('RGB')
        
        # Get label path
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.label_dir, f"{base_name}.txt")
        
        # Parse label file first to get original dimensions
        orig_width, orig_height = image.size
        boxes, labels = self.parse_label_file(label_path, orig_width, orig_height)
        
        # Apply transforms (this will convert to tensor)
        if self.split == 'train':
            image = self.train_transform(image)
        else:
            image = self.base_transform(image)
        
        # Convert to tensor
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }
        
        return image, target
    
    def parse_label_file(self, label_path, img_width, img_height):
        """Parse YOLO format label file."""
        boxes = []
        labels = []
        
        if not os.path.exists(label_path):
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.int64)
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                # YOLO format: class_id x_center y_center width height (normalized)
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # Convert to (x1, y1, x2, y2)
                x1 = max(0, x_center - width / 2)
                y1 = max(0, y_center - height / 2)
                x2 = min(img_width, x_center + width / 2)
                y2 = min(img_height, y_center + height / 2)
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)
        
        if not boxes:  # If no valid boxes, return empty tensors
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.int64)
            
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

def collate_fn(batch):
    """Collate function for DataLoader to handle custom batch formatting."""
    return tuple(zip(*batch))

def get_ip102_dataloaders(data_dir, batch_size=4, num_workers=4):
    """
    Create train and validation dataloaders for IP102 dataset.
    
    Args:
        data_dir: Path to the dataset root directory
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader: DataLoader instances for training and validation
    """
    # Create datasets
    train_dataset = IP102Dataset(data_dir, split='train')
    val_dataset = IP102Dataset(data_dir, split='val')
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the dataset
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Create dataset
    data_dir = "data/ip102/IP102_YOLOv5"
    dataset = IP102Dataset(data_dir, split='train')
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Visualize a few samples
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i in range(4):
        # Get the sample
        img_tensor, target = dataset[i]
        
        # Convert tensor to numpy and change from CxHxW to HxWxC
        img_np = img_tensor.numpy().transpose(1, 2, 0)
        
        # If the image was normalized, denormalize it
        if img_np.min() < 0 or img_np.max() > 1.0:
            img_np = (img_np * 0.5) + 0.5  # Assuming normalization with mean=0.5, std=0.5
            img_np = np.clip(img_np, 0, 1)
        
        # Plot the image
        axes[i].imshow(img_np)
        
        # Get the image dimensions after transforms
        img_height, img_width = img_np.shape[:2]
        
        # Add bounding boxes if any
        if len(target['boxes']) > 0:
            for box, label in zip(target['boxes'], target['labels']):
                # Convert box coordinates to integers
                x1, y1, x2, y2 = box.int().tolist()
                
                # Make sure coordinates are within image bounds
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(0, min(x2, img_width - 1))
                y2 = max(0, min(y2, img_height - 1))
                
                # Only draw if box is valid
                if x2 > x1 and y2 > y1:
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=1, edgecolor='r', facecolor='none'
                    )
                    axes[i].add_patch(rect)
                    
                    # Add class label
                    axes[i].text(
                        x1, y1, 
                        f'{dataset.classes[label]}', 
                        color='white', 
                        bbox=dict(facecolor='red', alpha=0.5),
                        fontsize=8
                    )
        
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')
    
    plt.tight_layout()
    plt.show()
    
    print("Dataset test completed successfully!")
