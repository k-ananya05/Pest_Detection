import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, img_size=640, split='train', transform=None):
        """
        YOLOv5 dataset format.
        
        Args:
            data_dir (str): Path to the dataset directory (should contain 'images' and 'labels' folders)
            img_size (int): Image size for resizing
            split (str): 'train' or 'val'
            transform: Optional transforms to be applied
        """
        super().__init__()
        self.img_dir = os.path.join(data_dir, 'images', split)
        self.label_dir = os.path.join(data_dir, 'labels', split)
        self.img_size = img_size
        self.transform = transform
        
        # Get list of image files
        self.img_files = [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(self.img_files)} images in {split} set")
        
        # Initialize class counts and max class ID
        self.class_counts = {}
        self.max_class_id = -1
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """Analyze the dataset to find class distribution and max class ID."""
        for img_name in self.img_files:
            label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            self.class_counts[class_id] = self.class_counts.get(class_id, 0) + 1
                            self.max_class_id = max(self.max_class_id, class_id)
        
        print(f"\nDataset contains {len(self.class_counts)} unique classes")
        print(f"Maximum class ID: {self.max_class_id}")
        print("Class distribution:")
        for class_id in sorted(self.class_counts.keys()):
            print(f"  Class {class_id}: {self.class_counts[class_id]} samples")
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        # Get corresponding label file
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')
        
        # Parse label file
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) < 5:  # Skip invalid lines
                        print(f"Warning: Invalid line in {label_path}, line {line_num}: {line.strip()}")
                        continue
                        
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        if class_id > self.max_class_id:
                            print(f"Warning: Class ID {class_id} exceeds maximum class ID {self.max_class_id} in {label_path}")
                            continue
                            
                        # Convert YOLO format (center_x, center_y, width, height) to (x1, y1, x2, y2)
                        x1 = (x_center - width / 2) * self.img_size
                        y1 = (y_center - height / 2) * self.img_size
                        x2 = (x_center + width / 2) * self.img_size
                        y2 = (y_center + height / 2) * self.img_size
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, min(x1, self.img_size - 1))
                        y1 = max(0, min(y1, self.img_size - 1))
                        x2 = max(0, min(x2, self.img_size - 1))
                        y2 = max(0, min(y2, self.img_size - 1))
                        
                        # Only add non-degenerate boxes
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(class_id + 1)  # Add 1 because class 0 is reserved for background
                            
                    except (ValueError, IndexError) as e:
                        print(f"Error processing line {line_num} in {label_path}: {e}")
                        continue
                    
                    # This block is now handled in the try-except above
        
        # Convert to tensors
        if self.transform:
            img = self.transform(img)
        
        # If no valid boxes, create a dummy box
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # (x2-x1) * (y2-y1)
        target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)  # all instances are not crowd
        
        return img, target

def collate_fn(batch):
    """Custom collate function for handling batches with different numbers of objects."""
    return tuple(zip(*batch))

def get_dataloaders(data_dir, batch_size=4, img_size=640):
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir (str): Path to the dataset directory
        batch_size (int): Batch size
        img_size (int): Image size for resizing
        
    Returns:
        train_loader, val_loader: DataLoader instances for training and validation
    """
    # Define transforms
    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = YOLODataset(data_dir, img_size=img_size, split='train', transform=train_transform)
    val_dataset = YOLODataset(data_dir, img_size=img_size, split='val', transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Temporarily set to 0 to avoid multiprocessing issues
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Temporarily set to 0 to avoid multiprocessing issues
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader

def visualize_sample(dataset, num_samples=4):
    """Visualize sample images with bounding boxes."""
    plt.figure(figsize=(15, 10))
    
    for i in range(min(num_samples, len(dataset))):
        img, target = dataset[i]
        
        # Convert tensor to PIL Image
        img = T.ToPILImage()(img)
        
        # Draw bounding boxes
        draw = ImageDraw.Draw(img)
        for box, label in zip(target['boxes'], target['labels']):
            # Convert from center_x, center_y, width, height to x1, y1, x2, y2
            x_center, y_center, width, height = box
            x1 = (x_center - width/2) * img.width
            y1 = (y_center - height/2) * img.height
            x2 = (x_center + width/2) * img.width
            y2 = (y_center + height/2) * img.height
            
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), f"Class {label}", fill="red")
        
        # Display image
        plt.subplot(2, 2, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image {i+1}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    data_dir = "data/ip102/IP102_YOLOv5"
    img_size = 640
    
    # Create dataset
    dataset = YOLODataset(data_dir, img_size=img_size, split='train')
    
    # Visualize samples
    print("Visualizing sample images...")
    visualize_sample(dataset, num_samples=4)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=4, img_size=img_size)
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
