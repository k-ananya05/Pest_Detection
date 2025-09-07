import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil
import argparse

from model import create_model
from yolo_dataset import get_dataloaders
from config import MODEL_CONFIG, TRAIN_CONFIG, PATHS, device

def train_one_epoch(model, optimizer, data_loader, epoch, writer=None):
    """Train the model for one epoch."""
    model.train()
    model.to(device)
    
    running_loss = 0.0
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        running_loss += losses.item()
        pbar.set_postfix(loss=running_loss/(batch_idx+1))
    
    avg_loss = running_loss / len(data_loader)
    
    if writer is not None:
        writer.add_scalar('Loss/train', avg_loss, epoch)
    
    return avg_loss

def evaluate(model, data_loader, device):
    """Evaluate the model on the validation set."""
    model.eval()
    model.to(device)
    
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item()
    
    avg_loss = running_loss / len(data_loader)
    return avg_loss

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Save model checkpoint."""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')

def main(mode="full"):
    writer = SummaryWriter(f"runs/pest_detection_{mode}")
    
    model = create_model(num_classes=MODEL_CONFIG['num_classes'])
    model = model.to(device)
    
    data_dir = os.path.join(PATHS['data'], 'ip102', 'IP102_YOLOv5')
    
    # Quick vs Full mode settings
    if mode == "quick":
        num_epochs = 2
        train_loader, val_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=2,
            img_size=224,          # smaller images
            subset_train=100,      # only 100 samples
            subset_val=20          # only 20 samples
        )
    else:  # full training
        num_epochs = 15
        train_loader, val_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=TRAIN_CONFIG['batch_size'],
            img_size=MODEL_CONFIG['image_size'][0]
        )
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=TRAIN_CONFIG['learning_rate'],
        momentum=TRAIN_CONFIG['momentum'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        train_loss = train_one_epoch(model, optimizer, train_loader, epoch, writer)
        val_loss = evaluate(model, val_loader, device)
        lr_scheduler.step()
        
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, is_best)
    
    writer.close()
    print(f"Training complete! Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="full",
                        help="Choose quick for debug training or full for full dataset training")
    args = parser.parse_args()
    main(mode=args.mode)
