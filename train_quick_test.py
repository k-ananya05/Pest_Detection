import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import shutil
from datetime import datetime
import torchvision

from model import create_model
from ip102_dataset import IP102Dataset, get_ip102_dataloaders
from config import MODEL_CONFIG, TRAIN_CONFIG, PATHS

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def collate_fn(batch):
    """Collate function for DataLoader to handle custom batch formatting."""
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, epoch, writer=None):
    """Train the model for one epoch."""
    model.train()
    model.to(device)
    
    running_loss = 0.0
    
    # Convert to list and take first 10 batches for quick testing
    data_list = []
    for i, batch in enumerate(data_loader):
        if i >= 10:  # Only take first 10 batches
            break
        data_list.append(batch)
    
    # Create a new data loader with just these batches
    class QuickDataLoader:
        def __init__(self, data):
            self.data = data
        def __iter__(self):
            return iter(self.data)
        def __len__(self):
            return len(self.data)
    
    data_loader = QuickDataLoader(data_list)
    
    # Progress bar
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Move data to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Initialize losses as a float
        losses = 0.0
        
        # Handle different return types from the model
        if isinstance(loss_dict, dict):
            # If it's a dictionary of losses, sum them
            losses = sum(loss for loss in loss_dict.values())
        elif isinstance(loss_dict, (list, tuple)):
            for loss_item in loss_dict:
                if isinstance(loss_item, dict):
                    losses += sum(loss for loss in loss_item.values())
                elif torch.is_tensor(loss_item):
                    losses += loss_item.item()
        elif torch.is_tensor(loss_dict):
            losses = loss_dict.item()
        
        # Backward pass and optimize
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Update running loss
        running_loss += losses.item()
        
        # Update progress bar
        pbar.set_postfix(loss=running_loss/(batch_idx+1))
    
    # Calculate average loss for the epoch
    avg_loss = running_loss / len(data_loader)
    
    # Log to TensorBoard if writer is provided
    if writer is not None:
        writer.add_scalar('Loss/train', avg_loss, epoch)
    
    return avg_loss

def evaluate(model, data_loader, device, writer=None, epoch=0):
    """Evaluate the model on the validation set."""
    # IMPORTANT: keep model in train() so it returns loss_dict
    model.train()
    model.to(device)
    
    running_loss = 0.0
    
    # Convert to list and take first 5 batches for quick testing
    data_list = []
    for i, batch in enumerate(data_loader):
        if i >= 5:  # Only take first 5 batches
            break
        data_list.append(batch)
    
    class QuickDataLoader:
        def __init__(self, data):
            self.data = data
        def __iter__(self):
            return iter(self.data)
        def __len__(self):
            return len(self.data)
    
    data_loader = QuickDataLoader(data_list)
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validating"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass (loss_dict will exist in train mode)
            loss_dict = model(images, targets)
            
            total_loss = 0.0
            if isinstance(loss_dict, dict):
                for loss in loss_dict.values():
                    if loss is not None:
                        total_loss += loss.item() if loss.dim() == 0 else loss.sum().item()
            elif isinstance(loss_dict, (list, tuple)):
                for loss_item in loss_dict:
                    if isinstance(loss_item, dict):
                        for loss in loss_item.values():
                            if loss is not None:
                                total_loss += loss.item() if loss.dim() == 0 else loss.sum().item()
                    elif torch.is_tensor(loss_item):
                        total_loss += loss_item.item() if loss_item.dim() == 0 else loss_item.sum().item()
            elif torch.is_tensor(loss_dict):
                total_loss = loss_dict.item() if loss_dict.dim() == 0 else loss_dict.sum().item()
            
            running_loss += float(total_loss)
    
    avg_loss = running_loss / len(data_loader)
    
    if writer is not None:
        writer.add_scalar('Loss/val', avg_loss, epoch)
    
    return avg_loss

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Save model checkpoint."""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')

def main():
    print("Starting quick test training...")
    print(f"Using device: {device}")
    
    # Create model with the correct number of classes from the dataset
    temp_dataset = IP102Dataset(os.path.join("data", "ip102", "IP102_YOLOv5"), split='train')
    num_classes = temp_dataset.num_classes
    print(f"Dataset has {num_classes} classes")
    
    model = create_model(num_classes=num_classes, pretrained=True)
    model.to(device)
    
    print("Loading data...")
    data_dir = os.path.join("data", "ip102", "IP102_YOLOv5")
    
    train_dataset = IP102Dataset(data_dir, split='train', img_size=400)
    val_dataset = IP102Dataset(data_dir, split='val', img_size=400)
    
    # Small subsets for quick test
    train_dataset.img_files = train_dataset.img_files[:100]
    val_dataset.img_files = val_dataset.img_files[:20]
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    
    output_dir = os.path.join(PATHS['model_save'], 'quick_test')
    os.makedirs(output_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=output_dir)
    
    best_loss = float('inf')
    num_epochs = 2
    
    print("Starting training loop...")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, epoch, writer)
        lr_scheduler.step()
        
        val_loss = evaluate(model, val_loader, device, writer, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f}")
        
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=os.path.join(output_dir, 'checkpoint.pth'))
    
    print(f"Quick test complete. Best validation loss: {best_loss:.4f}")
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    writer.close()

if __name__ == "__main__":
    main()
