# Cat Classifier

A machine learning project to classify images of my two cats: Nova and Zidane ("Zizi").

## Project Structure

```
cat_classifier/
├── data/               # Data directory
│   ├── raw/           # Original, untagged images
│   ├── train/         # Training dataset
│   ├── val/           # Validation dataset
│   └── test/          # Test dataset
├── src/               # Source code
├── models/            # Trained models
├── notebooks/         # Jupyter notebooks for exploration
└── config/            # Configuration files
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Organization

1. Place all original cat images in `data/raw/`
2. Run the data preparation script to organize images into train/val/test sets
3. Images will be automatically tagged based on their filenames or metadata

## Training

To train the model:
```bash
python src/train.py
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pandas
- numpy
- matplotlib
- jupyter (for notebooks) 