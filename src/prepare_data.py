import os
import shutil
import random
from pathlib import Path
from PIL import Image
import pandas as pd

def create_dataset_structure():
    """Create the necessary directories for the dataset."""
    base_dirs = ['train', 'val', 'test']
    classes = ['zizi', 'nova']
    
    for base_dir in base_dirs:
        for class_name in classes:
            os.makedirs(f'data/{base_dir}/{class_name}', exist_ok=True)

def split_dataset(raw_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        raw_dir (str): Directory containing raw images
        train_ratio (float): Ratio of images for training
        val_ratio (float): Ratio of images for validation
    """
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif')
    all_images = [f for f in os.listdir(raw_dir) if f.lower().endswith(image_extensions)]
    
    # Shuffle the images
    random.shuffle(all_images)
    
    # Calculate split indices
    total_images = len(all_images)
    train_split = int(total_images * train_ratio)
    val_split = int(total_images * (train_ratio + val_ratio))
    
    # Split the images
    train_images = all_images[:train_split]
    val_images = all_images[train_split:val_split]
    test_images = all_images[val_split:]
    
    return train_images, val_images, test_images

def tag_and_move_images(raw_dir, images, target_dir, class_name):
    """
    Move images to their respective directories and create a CSV with tags.
    
    Args:
        raw_dir (str): Source directory
        images (list): List of image filenames
        target_dir (str): Target directory
        class_name (str): Class name (zizi or nova)
    """
    tags = []
    
    for img in images:
        src_path = os.path.join(raw_dir, img)
        dst_path = os.path.join(target_dir, class_name, img)
        
        # Move the image
        shutil.copy2(src_path, dst_path)
        
        # Add to tags
        tags.append({
            'filename': img,
            'class': class_name,
            'path': dst_path
        })
    
    return tags

def main():
    # Create dataset structure
    create_dataset_structure()
    
    # Get raw images directory
    raw_dir = 'data/raw'
    
    # Split dataset
    train_images, val_images, test_images = split_dataset(raw_dir)
    
    # Process images for each class
    all_tags = []
    
    for class_name in ['zizi', 'nova']:
        # Train set
        train_tags = tag_and_move_images(
            raw_dir, train_images, 'data/train', class_name
        )
        all_tags.extend(train_tags)
        
        # Validation set
        val_tags = tag_and_move_images(
            raw_dir, val_images, 'data/val', class_name
        )
        all_tags.extend(val_tags)
        
        # Test set
        test_tags = tag_and_move_images(
            raw_dir, test_images, 'data/test', class_name
        )
        all_tags.extend(test_tags)
    
    # Save tags to CSV
    df = pd.DataFrame(all_tags)
    df.to_csv('data/tags.csv', index=False)
    print("Dataset preparation complete!")

if __name__ == '__main__':
    main() 