import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

class PestDataset(Dataset):
    def __init__(self, dataset, transform=None, is_train=True):
        """
        Args:
            dataset: Hugging Face dataset
            transform: Optional transform to be applied on a sample
            is_train: If True, apply data augmentation
        """
        self.dataset = dataset
        self.is_train = is_train
        
        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
        ])
        
        # Data augmentation for training
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(800, scale=(0.8, 1.0)),
        ])
        
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert("RGB")
        boxes = item['objects']['bbox']
        labels = item['objects']['category']
        
        # Apply base transform
        image = self.base_transform(image)
        
        # Apply data augmentation if training
        if self.is_train and random.random() > 0.5:
            # Convert to PIL for augmentation
            image_pil = transforms.ToPILImage()(image)
            image_pil = self.train_transform(image_pil)
            image = transforms.ToTensor()(image_pil)
        
        # Convert boxes to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        return image, target

def collate_fn(batch):
    """
    Custom collate function to handle variable number of objects in images
    """
    return tuple(zip(*batch))

def get_dataloaders(train_data, test_data, batch_size=4):
    """
    Create train and test dataloaders
    """
    train_dataset = PestDataset(train_data, is_train=True)
    test_dataset = PestDataset(test_data, is_train=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, test_loader
