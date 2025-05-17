# Polyp Segmentation using Kvasir-SEG Dataset

This project implements deep learning-based polyp segmentation using the Kvasir-SEG dataset. The goal is to develop an automated system that can accurately identify and delineate polyps in colonoscopy images.

## Dataset

The Kvasir-SEG dataset contains 1000 high-resolution colonoscopy images paired with their corresponding manually annotated segmentation masks. The dataset is available at: https://datasets.simula.no/kvasir-seg/

## Project Structure

```
.
├── data/               # Dataset directory
├── notebooks/         # Jupyter notebooks for analysis and visualization
├── src/              # Source code
│   ├── models/       # Model architectures
│   ├── utils/        # Utility functions
│   └── train.py      # Training script
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Kvasir-SEG dataset and place it in the `data` directory.

## Usage

1. Data preparation and preprocessing:
```bash
python src/prepare_data.py
```

2. Training the model:
```bash
python src/train.py
```

## Model Architecture

The project implements and evaluates various deep learning architectures for polyp segmentation:
- U-Net
- Attention U-Net

## Evaluation Metrics

The models are evaluated using:
- Dice coefficient
- Intersection-over-Union (IoU)

## Citation

If you use this code or the Kvasir-SEG dataset, please cite:
```
@inproceedings{jha2020kvasir,
  title={Kvasir-seg: A segmented polyp dataset},
  author={Jha, Debesh and Smedsrud, Pia H and Riegler, Michael A and Halvorsen, P{\aa}l and de Lange, Thomas and Johansen, Dag and Johansen, H{\aa}vard D},
  booktitle={International Conference on Multimedia Modeling},
  pages={451--462},
  year={2020},
  organization={Springer}
}
``` 