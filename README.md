# Cat Classifier

A machine learning project to classify images of my two cats, Nova and Zidane ("Zizi").

## Project Disclaimer

This project was developed with the support of Cursor and serves as a learning opportunity for me to refresh my programming and machine learning skills after many years. The project documentation and code structure are designed to be educational and may include explanations of basic concepts as I re-learn them.

As someone who graduated in 2015 (back when we had to debug our code the old-fashioned way - by staring at it for hours), I figured I'd see what these AI coding assistants were all about. Turns out they're not terrible. It's kind of like having a TA who's always available and doesn't mind explaining things multiple times. While I'm still doing the actual work, it's been useful for getting back up to speed with modern ML concepts. At least now when I'm stuck, I can ask something that won't roll its eyes at me.

## Project Structure

```
cat_classifier/
├── data/
│   ├── raw/              # Original images
│   ├── train/            # Training set
│   │   ├── nova/         # Nova training images
│   │   └── zizi/         # Zizi training images
│   ├── val/              # Validation set
│   │   ├── nova/         # Nova validation images
│   │   └── zizi/         # Zizi validation images
│   └── test/             # Test set
│       ├── nova/         # Nova test images
│       └── zizi/         # Zizi test images
├── models/               # Saved model checkpoints
├── results/              # Evaluation results and visualizations
├── src/
│   ├── tag_images.py     # Image tagging helper script
│   ├── organize_dataset.py # Dataset organization script
│   ├── train.py          # Model training script
│   └── test_model.py     # Model testing script
└── requirements.txt      # Project dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your cat images in the `data/raw` directory.

## Usage

### 1. Tag Images
Use the tagging script to label your cat images:
```bash
python src/tag_images.py
```
Controls:
- `z`: Tag as Zizi
- `n`: Tag as Nova
- `s`: Skip image
- `c`: Crop image (for images with both cats)
- `q`: Quit tagging

### 2. Organize Dataset
After tagging, organize the images into training, validation, and test sets:
```bash
python src/organize_dataset.py
```
This will:
- Split images into 70% training, 15% validation, and 15% test sets
- Create the necessary directory structure
- Save dataset information to `data/dataset_info.json`

### 3. Train Model
Train the cat classification model:
```bash
python src/train.py
```
The script will:
- Use a pre-trained ResNet18 model
- Fine-tune for cat classification
- Save the best model to `models/best_model.pth`
- Implement early stopping to prevent overfitting

### 4. Test Model
Evaluate the model's performance:
```bash
python src/test_model.py
```
This will:
- Generate a classification report
- Create a confusion matrix
- Show sample predictions with confidence scores
- Save visualizations to the `results` directory

## Current Performance

The model achieves:
- Overall accuracy: 69%
- Balanced performance between classes:
  - Nova: 71% precision, 67% recall
  - Zizi: 67% precision, 71% recall

## Future Improvements

1. Data Augmentation:
   - Add more training data
   - Apply image transformations
   - Include more variations in lighting and angles

2. Model Architecture:
   - Try different pre-trained models
   - Adjust learning rate and training parameters
   - Implement regularization techniques

3. Training Process:
   - Increase training epochs
   - Add learning rate scheduling
   - Implement more sophisticated data augmentation

## Dependencies

- torch>=2.0.0
- torchvision>=0.15.0
- numpy>=1.21.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- jupyter>=1.0.0
- pillow>=8.3.0
- scikit-learn>=0.24.0
- tqdm>=4.62.0
- opencv-python>=4.5.0
- seaborn>=0.12.0 