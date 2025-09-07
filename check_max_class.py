import os

def find_max_class_id(data_dir):
    max_class = -1
    
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
                        max_class = max(max_class, class_id)
    
    return max_class

if __name__ == "__main__":
    data_dir = "data/ip102/IP102_YOLOv5"
    max_class = find_max_class_id(data_dir)
    print(f"Maximum class ID found: {max_class}")
    print(f"Update config.py with: \"num_classes\": {max_class + 2}  # {max_class + 1} classes + 1 for background")
