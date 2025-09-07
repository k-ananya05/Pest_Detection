import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import shutil
from datetime import datetime

from model import create_model
from ip102_dataset import IP102Dataset, get_ip102_dataloaders
from config import MODEL_CONFIG, TRAIN_CONFIG, PATHS

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_one_epoch(model, optimizer, data_loader, epoch, writer=None):
    """Train the model for one epoch."""
    model.train()
    model.to(device)
    
    running_loss = 0.0
    
    # Progress bar
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Move data to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
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
    model.eval()
    model.to(device)
    
    running_loss = 0.0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validating"):
            # Move data to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item()
    
    avg_loss = running_loss / len(data_loader)
    
    # Log to TensorBoard if writer is provided
    if writer is not None:
        writer.add_scalar('Loss/val', avg_loss, epoch)
    
    return avg_loss

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Save model checkpoint."""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')

def main():
    print("Starting training...")
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(num_classes=102)  # IP102 has 102 classes
    
    # Move model to device
    model.to(device)
    
    # Create data loaders
    print("Loading data...")
    data_dir = os.path.join("data", "ip102", "IP102_YOLOv5")
    
    # Create data loaders with appropriate batch size
    train_loader, val_loader = get_ip102_dataloaders(
        data_dir=data_dir,
        batch_size=TRAIN_CONFIG['batch_size'],
        num_workers=4
    )
    
    # Define optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=TRAIN_CONFIG['learning_rate'],
        momentum=TRAIN_CONFIG['momentum'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Create output directory
    output_dir = os.path.join(PATHS['model_save'], datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_dir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=output_dir)
    
    # Training loop
    best_loss = float('inf')
    
    print("Starting training loop...")
    for epoch in range(TRAIN_CONFIG['num_epochs']):
        # Train for one epoch
        train_loss = train_one_epoch(model, optimizer, train_loader, epoch, writer)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        val_loss = evaluate(model, val_loader, device, writer, epoch)
        
        # Print progress
        print(f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=os.path.join(output_dir, 'checkpoint.pth'))
    
    print(f"Training complete. Best validation loss: {best_loss:.4f}")
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    
    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
