import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.data_utils import PolypDataset, get_transforms
import numpy as np
from tqdm import tqdm
import os
import argparse
import csv
from models.unet import UNet

def dice_coef(preds, targets, threshold=0.5, eps=1e-6):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean().item()

def iou_coef(preds, targets, threshold=0.5, eps=1e-6):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = (preds + targets).clamp(0,1).sum(dim=(1,2,3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        bce_loss = self.bce(preds, targets)
        smooth = 1e-6
        intersection = (preds * targets).sum(dim=(1,2,3))
        union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
        dice_loss = 1 - ((2. * intersection + smooth) / (union + smooth))
        return bce_loss + dice_loss.mean()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    dice_scores = []
    iou_scores = []
    for images, masks in tqdm(loader, desc='Train', leave=False):
        images = images.to(device)
        masks = masks.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        preds = torch.sigmoid(outputs)
        dice_scores.append(dice_coef(preds, masks))
        iou_scores.append(iou_coef(preds, masks))
    return epoch_loss / len(loader), np.mean(dice_scores), np.mean(iou_scores)

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    dice_scores = []
    iou_scores = []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Val', leave=False):
            images = images.to(device)
            masks = masks.float().unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            epoch_loss += loss.item()
            preds = torch.sigmoid(outputs)
            dice_scores.append(dice_coef(preds, masks))
            iou_scores.append(iou_coef(preds, masks))
    return epoch_loss / len(loader), np.mean(dice_scores), np.mean(iou_scores)

def get_model(model_name):
    if model_name == 'unet':
        return UNet(in_channels=3, out_channels=1)
    elif model_name == 'attention_unet':
        from models.attention_unet import AttentionUNet
        return AttentionUNet(in_channels=3, out_channels=1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_dice = 0
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            train_loss += loss.item()
            train_dice += dice_coef(outputs, masks)
        
        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice_coef(outputs, masks)
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        # Print metrics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Best model saved!')

def dice_coefficient(pred, target):
    smooth = 1e-5
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'attention_unet'], help='Model to use')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = PolypDataset('data/Kvasir-SEG/train', get_transforms(is_train=True), is_train=True)
    val_dataset = PolypDataset('data/Kvasir-SEG/val', get_transforms(is_train=False), is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = get_model(args.model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceBCELoss()
    best_val_dice = 0
    os.makedirs('checkpoints', exist_ok=True)
    results = []
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = eval_one_epoch(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")
        results.append([epoch, train_loss, train_dice, train_iou, val_loss, val_dice, val_iou])
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), f'checkpoints/best_{args.model}.pth')
            print("Best model saved!")
    # Save results to CSV
    csv_path = f'results_{args.model}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_dice', 'train_iou', 'val_loss', 'val_dice', 'val_iou'])
        writer.writerows(results)
    print(f"Results saved to {csv_path}")

if __name__ == '__main__':
    main() 