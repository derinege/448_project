import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Define paths
base_path = Path('data/Kvasir-SEG')
images_path = base_path / 'images'
masks_path = base_path / 'masks'

# Create train, val, test directories
for split in ['train', 'val', 'test']:
    for subdir in ['images', 'masks']:
        os.makedirs(base_path / split / subdir, exist_ok=True)

# Get all image files
image_files = sorted([f for f in os.listdir(images_path) if f.endswith('.jpg')])
total_images = len(image_files)

# Calculate split sizes
train_size = int(0.7 * total_images)
val_size = int(0.15 * total_images)
test_size = total_images - train_size - val_size

# Shuffle files
random.shuffle(image_files)

# Split files
train_files = image_files[:train_size]
val_files = image_files[train_size:train_size + val_size]
test_files = image_files[train_size + val_size:]

# Function to copy files
def copy_files(file_list, split_name):
    for img_file in file_list:
        # Copy image
        shutil.copy2(
            images_path / img_file,
            base_path / split_name / 'images' / img_file
        )
        # Copy corresponding mask
        shutil.copy2(
            masks_path / img_file,
            base_path / split_name / 'masks' / img_file
        )

# Copy files to respective directories
copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

# Print statistics
print(f"Total images: {total_images}")
print(f"Train set: {len(train_files)} images")
print(f"Validation set: {len(val_files)} images")
print(f"Test set: {len(test_files)} images") 