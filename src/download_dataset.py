import os
import gdown
import zipfile
from pathlib import Path

def download_dataset():
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Download Kvasir-SEG dataset
    print("Downloading Kvasir-SEG dataset...")
    url = 'https://drive.google.com/uc?id=1Jt6vfmmjXf9sp8jw1rB48P06v0gHz5ti'
    output = 'data/Kvasir-SEG.zip'
    gdown.download(url, output, quiet=False)
    
    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('data')
    
    # Remove zip file
    os.remove(output)
    
    print("Dataset downloaded and extracted successfully!")
    print("Running prepare_data.py to split the dataset...")
    
    # Run prepare_data.py to split the dataset
    from prepare_data import main as prepare_data
    prepare_data()

if __name__ == '__main__':
    download_dataset() 