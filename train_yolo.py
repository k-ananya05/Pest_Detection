import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil

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
        images = [image.to(device) for image in images]
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


def evaluate(model, data_loader, device, writer=None, epoch=None):
    """Evaluate the model on the validation set."""
    model.eval()  # <-- use eval mode
    model.to(device)

    running_loss = 0.0
    total_preds, total_matches = 0, 0  # For IoU-based accuracy

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validating"):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Compute validation loss
            with torch.no_grad():
                # Forward pass with targets to get losses
                loss_dict = model(images, targets)
                
                # Handle both dictionary and list outputs
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values())
                elif isinstance(loss_dict, (list, tuple)):
                    # If model returns a list of losses, sum them
                    losses = sum(loss_dict)
                else:
                    # If it's a single tensor
                    losses = loss_dict
                    
                running_loss += losses.item()

            # Get predictions (without computing gradients)
            with torch.no_grad():
                predictions = model(images)

            # Very simple accuracy: IoU >= 0.5 counts as match
            for pred, tgt in zip(predictions, targets):
                if "boxes" not in pred or len(pred["boxes"]) == 0:
                    continue
                if len(tgt["boxes"]) == 0:
                    continue

                pred_boxes = pred["boxes"].cpu()
                tgt_boxes = tgt["boxes"].cpu()

                for pb in pred_boxes:
                    total_preds += 1
                    ious = box_iou(pb.unsqueeze(0), tgt_boxes)
                    if torch.any(ious >= 0.5):
                        total_matches += 1

    avg_loss = running_loss / len(data_loader)
    acc = total_matches / total_preds if total_preds > 0 else 0.0

    if writer is not None and epoch is not None:
        writer.add_scalar('Loss/val', avg_loss, epoch)
        writer.add_scalar('Accuracy/val', acc, epoch)

    return avg_loss, acc


def box_iou(box1, box2):
    """Compute IoU between two sets of boxes."""
    # box format: [x1, y1, x2, y2]
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # top-left
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # bottom-right

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Save model checkpoint."""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


def main():
    writer = SummaryWriter('runs/pest_detection')

    model = create_model(num_classes=MODEL_CONFIG['num_classes'])
    model = model.to(device)

    data_dir = os.path.join(PATHS['data'], 'ip102', 'IP102_YOLOv5')
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

    for epoch in range(TRAIN_CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}")
        print("-" * 10)

        train_loss = train_one_epoch(model, optimizer, train_loader, epoch, writer)
        val_loss, val_acc = evaluate(model, val_loader, device, writer, epoch)

        lr_scheduler.step()

        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, is_best)

    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
