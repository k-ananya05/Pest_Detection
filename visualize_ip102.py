import os
import random
import yaml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

class IP102Visualizer:
    def __init__(self, data_dir):
        """Initialize the IP102 dataset visualizer."""
        self.data_dir = os.path.join(data_dir, 'IP102_YOLOv5')
        self.data_yaml = os.path.join(self.data_dir, 'ip102.yaml')
        
        # Load dataset information
        with open(self.data_yaml) as f:
            self.data = yaml.safe_load(f)
            
        # Set up paths
        self.train_img_dir = os.path.join(self.data_dir, 'images', 'train')
        self.val_img_dir = os.path.join(self.data_dir, 'images', 'val')
        self.train_label_dir = os.path.join(self.data_dir, 'labels', 'train')
        self.val_label_dir = os.path.join(self.data_dir, 'labels', 'val')
        
        # Get class names
        self.classes = self.data['names']
        self.nc = len(self.classes)
        
        # Get list of image files
        self.train_images = [os.path.join(self.train_img_dir, f) for f in os.listdir(self.train_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.val_images = [os.path.join(self.val_img_dir, f) for f in os.listdir(self.val_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Found {len(self.train_images)} training images")
        print(f"Found {len(self.val_images)} validation images")
        print(f"Number of classes: {self.nc}")
    
    def parse_label_file(self, label_path, img_width, img_height):
        """Parse YOLO format label file and return boxes and class IDs."""
        boxes = []
        class_ids = []
        
        if not os.path.exists(label_path):
            return boxes, class_ids
            
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # Convert from center to top-left coordinates
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                boxes.append([x1, y1, x2, y2])
                class_ids.append(class_id)
                
        return boxes, class_ids
    
    def draw_boxes(self, image, boxes, class_ids):
        """Draw bounding boxes on the image."""
        draw = ImageDraw.Draw(image)
        
        for box, class_id in zip(boxes, class_ids):
            x1, y1, x2, y2 = box
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            
            # Draw label background
            label = f"{self.classes[class_id]} ({class_id})"
            text_bbox = draw.textbbox((x1, y1 - 15), label)
            draw.rectangle(text_bbox, fill="red")
            
            # Draw label text
            draw.text((x1, y1 - 15), label, fill="white")
            
        return image
    
    def visualize_random_samples(self, split='train', num_samples=5):
        """Visualize random samples from the dataset."""
        if split == 'train':
            img_dir = self.train_img_dir
            label_dir = self.train_label_dir
            images = self.train_images
        else:
            img_dir = self.val_img_dir
            label_dir = self.val_label_dir
            images = self.val_images
            
        # Randomly select samples
        selected = random.sample(images, min(num_samples, len(images)))
        
        # Create a figure
        fig, axes = plt.subplots(1, len(selected), figsize=(20, 4))
        if len(selected) == 1:
            axes = [axes]
            
        for i, img_path in enumerate(selected):
            # Load image
            img = Image.open(img_path)
            
            # Get corresponding label file
            label_path = os.path.join(label_dir, os.path.basename(img_path).split('.')[0] + '.txt')
            
            # Parse label file
            boxes, class_ids = self.parse_label_file(label_path, img.width, img.height)
            
            # Draw boxes
            img_with_boxes = self.draw_boxes(img.copy(), boxes, class_ids)
            
            # Display image
            axes[i].imshow(img_with_boxes)
            axes[i].set_title(f"{split.capitalize()} Sample {i+1}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Initialize visualizer
    visualizer = IP102Visualizer("data/ip102")
    
    # Visualize training samples
    print("\nVisualizing training samples...")
    visualizer.visualize_random_samples(split='train')
    
    # Visualize validation samples
    print("\nVisualizing validation samples...")
    visualizer.visualize_random_samples(split='valid')
