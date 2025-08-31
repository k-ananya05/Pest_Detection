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
from dataset import get_dataloaders
from config import MODEL_CONFIG, TRAIN_CONFIG, PATHS, device

def train_one_epoch(model, optimizer, data_loader, epoch, writer=None):
    """Train the model for one epoch."""
    model.train()
    model.to(device)
    
    running_loss = 0.0
    
    # Progress bar
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Move data to device
        images = [img.to(device) for img in images]
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
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
    
    return avg_loss

def evaluate(model, data_loader, device):
    """Evaluate the model on the validation set."""
    model.eval()
    model.to(device)
    
    running_loss = 0.0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Update running loss
            running_loss += losses.item()
    
    # Calculate average loss
    avg_loss = running_loss / len(data_loader)
    
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
    model = create_model(num_classes=MODEL_CONFIG['num_classes'])
    
    # Move model to device
    model.to(device)
    
    # Create data loaders
    print("Loading data...")
    from datasets import load_dataset
    dataset = load_dataset("wadhwani-ai/pest-management-opendata", streaming=True)
    
    # Take a subset for training (you can adjust the number of samples)
    train_data = dataset['train'].shuffle(seed=42).take(1000)
    test_data = dataset['test'].take(200)
    
    train_loader, test_loader = get_dataloaders(
        train_data, 
        test_data, 
        batch_size=TRAIN_CONFIG['batch_size']
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
    output_dir = PATHS['model_save'] / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        val_loss = evaluate(model, test_loader, device)
        
        # Print progress
        print(f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f}")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Save checkpoint
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=output_dir / 'checkpoint.pth')
    
    print(f"Training complete. Best validation loss: {best_loss:.4f}")
    
    # Save the final model
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    
    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
