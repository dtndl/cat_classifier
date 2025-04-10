import json
import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def load_tags(tags_file):
    """Load tags from JSON file."""
    with open(tags_file, 'r') as f:
        return json.load(f)

def get_class_counts(tags):
    """Count images per class."""
    counts = {'zizi': 0, 'nova': 0}
    for tag_info in tags.values():
        if tag_info['tag'] in counts:
            counts[tag_info['tag']] += 1
    return counts

def split_dataset(tags, train_ratio=0.7, val_ratio=0.15):
    """
    Split tagged images into train, validation, and test sets.
    
    Args:
        tags (dict): Dictionary of image tags
        train_ratio (float): Ratio of images for training
        val_ratio (float): Ratio of images for validation
    """
    # Filter out skipped images and get lists per class
    zizi_images = [img for img, tag_info in tags.items() if tag_info['tag'] == 'zizi']
    nova_images = [img for img, tag_info in tags.items() if tag_info['tag'] == 'nova']
    
    # Shuffle the lists
    random.shuffle(zizi_images)
    random.shuffle(nova_images)
    
    # Calculate split indices for each class
    zizi_train_split = int(len(zizi_images) * train_ratio)
    zizi_val_split = int(len(zizi_images) * (train_ratio + val_ratio))
    
    nova_train_split = int(len(nova_images) * train_ratio)
    nova_val_split = int(len(nova_images) * (train_ratio + val_ratio))
    
    # Split the images
    splits = {
        'train': {
            'zizi': zizi_images[:zizi_train_split],
            'nova': nova_images[:nova_train_split]
        },
        'val': {
            'zizi': zizi_images[zizi_train_split:zizi_val_split],
            'nova': nova_images[nova_train_split:nova_val_split]
        },
        'test': {
            'zizi': zizi_images[zizi_val_split:],
            'nova': nova_images[nova_val_split:]
        }
    }
    
    return splits

def copy_images(splits, output_dir):
    """Copy images to their respective directories."""
    for split_name, classes in splits.items():
        for class_name, images in classes.items():
            # Create directory if it doesn't exist
            target_dir = Path(output_dir) / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            for img_path in tqdm(images, desc=f"Copying {class_name} images to {split_name}"):
                src_path = Path(img_path)
                dst_path = target_dir / src_path.name
                shutil.copy2(src_path, dst_path)

def create_dataset_info(splits, output_dir):
    """Create a JSON file with dataset information."""
    info = {
        'total_images': sum(len(imgs) for split in splits.values() for imgs in split.values()),
        'splits': {
            split_name: {
                class_name: len(images)
                for class_name, images in classes.items()
            }
            for split_name, classes in splits.items()
        }
    }
    
    with open(Path(output_dir) / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)

def main():
    # Configuration
    tags_file = 'data/image_tags.json'
    output_dir = 'data'
    
    # Load tags
    print("Loading tags...")
    tags = load_tags(tags_file)
    
    # Show class counts
    counts = get_class_counts(tags)
    print("\nClass counts:")
    for class_name, count in counts.items():
        print(f"{class_name}: {count} images")
    
    # Split dataset
    print("\nSplitting dataset...")
    splits = split_dataset(tags)
    
    # Copy images
    print("\nOrganizing images...")
    copy_images(splits, output_dir)
    
    # Create dataset info
    create_dataset_info(splits, output_dir)
    
    print("\nDataset organization complete!")
    print("Images have been organized into:")
    print("- data/train/zizi/ and data/train/nova/")
    print("- data/val/zizi/ and data/val/nova/")
    print("- data/test/zizi/ and data/test/nova/")
    print("\nDataset information has been saved to data/dataset_info.json")

if __name__ == '__main__':
    main() 