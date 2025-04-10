import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import json
from datetime import datetime

class ImageTagger:
    def __init__(self, raw_dir='data/raw', output_dir='data'):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.tags_file = self.output_dir / 'image_tags.json'
        
        # Create necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'train').mkdir(exist_ok=True)
        (self.output_dir / 'val').mkdir(exist_ok=True)
        (self.output_dir / 'test').mkdir(exist_ok=True)
        
        # Load existing tags if any
        self.tags = self.load_tags()
        
        # Get list of images to process
        self.images = self.get_images_to_process()
        
        # Current image index
        self.current_idx = 0
        
    def load_tags(self):
        """Load existing tags from JSON file."""
        if self.tags_file.exists():
            with open(self.tags_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_tags(self):
        """Save tags to JSON file."""
        with open(self.tags_file, 'w') as f:
            json.dump(self.tags, f, indent=2)
    
    def get_images_to_process(self):
        """Get list of images that haven't been tagged yet."""
        all_images = [f for f in self.raw_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        return [img for img in all_images if str(img) not in self.tags]
    
    def show_image(self, img_path):
        """Display image with controls."""
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not load image: {img_path}")
            return False
        
        # Resize image if too large
        max_height = 800
        if img.shape[0] > max_height:
            scale = max_height / img.shape[0]
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        # Add instructions to image
        instructions = [
            "Controls:",
            "z - Tag as Zizi",
            "n - Tag as Nova",
            "s - Skip this image",
            "c - Crop image",
            "q - Quit"
        ]
        
        # Create a black bar for instructions
        bar_height = len(instructions) * 30 + 20
        bar = np.zeros((bar_height, img.shape[1], 3), dtype=np.uint8)
        
        # Add instructions to the bar
        for i, text in enumerate(instructions):
            cv2.putText(bar, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
        
        # Combine image and instructions
        combined = np.vstack((img, bar))
        
        cv2.imshow('Image Tagger', combined)
        return True
    
    def crop_image(self, img_path):
        """Allow user to crop the image."""
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        # Create window and set mouse callback
        cv2.namedWindow('Crop Image')
        cv2.setMouseCallback('Crop Image', self.mouse_callback)
        
        self.cropping = False
        self.start_x, self.start_y = -1, -1
        self.end_x, self.end_y = -1, -1
        
        clone = img.copy()
        
        while True:
            display = img.copy()
            if self.cropping:
                cv2.rectangle(display, (self.start_x, self.start_y), 
                            (self.end_x, self.end_y), (0, 255, 0), 2)
            
            cv2.imshow('Crop Image', display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # Start cropping
                self.cropping = True
                self.start_x, self.start_y = self.end_x, self.end_y
            elif key == ord('s'):  # Save crop
                if self.start_x != -1 and self.end_x != -1:
                    x1, x2 = sorted([self.start_x, self.end_x])
                    y1, y2 = sorted([self.start_y, self.end_y])
                    cropped = img[y1:y2, x1:x2]
                    return cropped
            elif key == ord('q'):  # Quit cropping
                return None
        
        cv2.destroyWindow('Crop Image')
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for cropping."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_x, self.start_y = x, y
            self.cropping = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping:
                self.end_x, self.end_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.end_x, self.end_y = x, y
            self.cropping = False
    
    def process_image(self):
        """Process current image and get user input."""
        if self.current_idx >= len(self.images):
            return False
        
        img_path = self.images[self.current_idx]
        if not self.show_image(img_path):
            self.current_idx += 1
            return True
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('z'):  # Tag as Zizi
                self.tags[str(img_path)] = {'tag': 'zizi', 'timestamp': datetime.now().isoformat()}
                self.save_tags()
                self.current_idx += 1
                break
            elif key == ord('n'):  # Tag as Nova
                self.tags[str(img_path)] = {'tag': 'nova', 'timestamp': datetime.now().isoformat()}
                self.save_tags()
                self.current_idx += 1
                break
            elif key == ord('s'):  # Skip
                self.tags[str(img_path)] = {'tag': 'skip', 'timestamp': datetime.now().isoformat()}
                self.save_tags()
                self.current_idx += 1
                break
            elif key == ord('c'):  # Crop
                cropped = self.crop_image(img_path)
                if cropped is not None:
                    # Save cropped image
                    cropped_path = img_path.parent / f"{img_path.stem}_cropped{img_path.suffix}"
                    cv2.imwrite(str(cropped_path), cropped)
                    self.images.insert(self.current_idx + 1, cropped_path)
                cv2.destroyWindow('Crop Image')
                self.show_image(img_path)
            elif key == ord('q'):  # Quit
                return False
        
        cv2.destroyAllWindows()
        return True
    
    def run(self):
        """Run the image tagging process."""
        print(f"Found {len(self.images)} images to process.")
        print("Controls:")
        print("z - Tag as Zizi")
        print("n - Tag as Nova")
        print("s - Skip this image")
        print("c - Crop image")
        print("q - Quit")
        
        while self.process_image():
            pass
        
        print("\nTagging complete!")
        print(f"Processed {self.current_idx} images.")
        print(f"Tags saved to {self.tags_file}")

def main():
    tagger = ImageTagger()
    tagger.run()

if __name__ == '__main__':
    main() 