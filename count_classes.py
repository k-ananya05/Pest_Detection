import os
from collections import defaultdict

def count_classes(data_dir):
    """Count the number of unique classes in the dataset."""
    class_counts = defaultdict(int)
    
    # Check train and val sets
    for split in ['train', 'val']:
        label_dir = os.path.join(data_dir, 'labels', split)
        if not os.path.exists(label_dir):
            print(f"Warning: {label_dir} does not exist")
            continue
            
        for label_file in os.listdir(label_dir):
            if not label_file.endswith('.txt'):
                continue
                
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:  # At least class_id
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
    
    print("\nClass distribution:")
    for class_id in sorted(class_counts.keys()):
        print(f"Class {class_id}: {class_counts[class_id]} samples")
    
    print(f"\nTotal unique classes: {len(class_counts)}")
    return len(class_counts)

if __name__ == "__main__":
    data_dir = "data/ip102/IP102_YOLOv5"
    num_classes = count_classes(data_dir)
    print(f"\nUpdate config.py with: \"num_classes\": {num_classes + 1}  # +1 for background class")
    print("Note: In the YOLO dataset, class IDs should be 0-based (0 to num_classes-1)")
