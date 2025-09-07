import os
import shutil
import zipfile
import requests
from pathlib import Path
import yaml
from tqdm import tqdm

def download_file(url, filename):
    """Download a file with progress bar."""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def setup_ip102_dataset():
    """Setup IP102 dataset from Kaggle."""
    # Create data directory
    data_dir = Path("data/ip102")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset (you'll need to download this manually from Kaggle first)
    # Note: You need to accept the competition rules on Kaggle first
    kaggle_url = "https://www.kaggle.com/datasets/leonidkulyk/ip102-yolov5"
    print(f"Please download the dataset manually from: {kaggle_url}")
    print("After downloading, place the zip file in the 'data/ip102' directory.")
    
    # Expected zip file name (update this if the filename is different)
    zip_path = data_dir / "ip102-yolov5.zip"
    
    # Check if zip file exists
    if not zip_path.exists():
        print(f"\nError: Could not find {zip_path}")
        print(f"Please download the dataset from {kaggle_url} and place it in {data_dir}")
        return False
    
    # Extract dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    print("\nDataset structure:")
    print(f"{data_dir}/")
    print("├── data.yaml")
    print("├── train/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("├── valid/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("└── test/")
    print("    ├── images/")
    print("    └── labels/")
    
    # Load and display dataset information
    try:
        with open(data_dir / 'data.yaml') as f:
            data = yaml.safe_load(f)
            print("\nDataset information:")
            print(f"Number of classes: {data['nc']}")
            print(f"Class names: {data['names']}")
            print(f"Number of training images: {len(list((data_dir / 'train' / 'images').glob('*')))}")
            print(f"Number of validation images: {len(list((data_dir / 'valid' / 'images').glob('*')))}")
            print(f"Number of test images: {len(list((data_dir / 'test' / 'images').glob('*')))}")
    except Exception as e:
        print(f"Error reading dataset info: {e}")
    
    return True

if __name__ == "__main__":
    setup_ip102_dataset()
