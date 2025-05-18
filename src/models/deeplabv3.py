import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.model = deeplabv3_resnet50(pretrained=pretrained)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        out = self.model(x)['out']
        return out  # Sigmoid veya BCEWithLogitsLoss ile uyumlu