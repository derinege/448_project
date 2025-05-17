import matplotlib.pyplot as plt
from utils.data_utils import PolypDataset, get_transforms
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

def main():
    # Create datasets
    train_dataset = PolypDataset(
        data_dir='data/Kvasir-SEG/train',
        transform=get_transforms(is_train=True),
        is_train=True
    )

    # Create data loader with num_workers=0 for testing
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # Set to 0 for testing
    )

    # Get a batch of images
    images, masks = next(iter(train_loader))

    # Plot images and masks
    plt.figure(figsize=(15, 5))
    for i in range(4):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).numpy())
        plt.title(f'Image {i}')
        plt.axis('off')
        
        plt.subplot(2, 4, i + 5)
        plt.imshow(masks[i].numpy(), cmap='gray')
        plt.title(f'Mask {i}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('data_loader_test.png')
    plt.close()

if __name__ == '__main__':
    mp.freeze_support()
    main() 