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

def pixel_accuracy(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    correct = (preds == targets).float().sum()
    total = torch.numel(preds)
    return (correct / total).item()

def precision_recall(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return precision.item(), recall.item()

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
    dice_scores, iou_scores, acc_scores, prec_scores, rec_scores = [], [], [], [], []
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
        acc_scores.append(pixel_accuracy(preds, masks))
        prec, rec = precision_recall(preds, masks)
        prec_scores.append(prec)
        rec_scores.append(rec)
    return (epoch_loss / len(loader),
            np.mean(dice_scores),
            np.mean(iou_scores),
            np.mean(acc_scores),
            np.mean(prec_scores),
            np.mean(rec_scores))

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    dice_scores, iou_scores, acc_scores, prec_scores, rec_scores = [], [], [], [], []
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
            acc_scores.append(pixel_accuracy(preds, masks))
            prec, rec = precision_recall(preds, masks)
            prec_scores.append(prec)
            rec_scores.append(rec)
    return (epoch_loss / len(loader),
            np.mean(dice_scores),
            np.mean(iou_scores),
            np.mean(acc_scores),
            np.mean(prec_scores),
            np.mean(rec_scores))

def get_model(model_name):
    if model_name == 'unet':
        from models.unet import UNet
        return UNet(in_channels=3, out_channels=1)
    elif model_name == 'attention_unet':
        from models.attention_unet import AttentionUNet
        return AttentionUNet(in_channels=3, out_channels=1)
    elif model_name == 'deeplabv3':
        from models.deeplabv3 import DeepLabV3Plus
        return DeepLabV3Plus(num_classes=1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'attention_unet', 'deeplabv3'], help='Model to use')
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
        train_loss, train_dice, train_iou, train_acc, train_prec, train_rec = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou, val_acc, val_prec, val_rec = eval_one_epoch(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f} | Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")
        results.append([epoch, train_loss, train_dice, train_iou, train_acc, train_prec, train_rec,
                        val_loss, val_dice, val_iou, val_acc, val_prec, val_rec])
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), f'checkpoints/best_{args.model}.pth')
            print("Best model saved!")
    # Save results to CSV
    csv_path = f'results_{args.model}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_dice', 'train_iou', 'train_acc', 'train_prec', 'train_rec',
                         'val_loss', 'val_dice', 'val_iou', 'val_acc', 'val_prec', 'val_rec'])
        writer.writerows(results)
    print(f"Results saved to {csv_path}")

if __name__ == '__main__':
    main()