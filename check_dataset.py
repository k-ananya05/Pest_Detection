from datasets import load_dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import random
import numpy as np

def load_pest_dataset():
    """Load a plant disease dataset from Hugging Face."""
    print("Loading Cassava Leaf Disease dataset...")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("cassava-leaf-disease", split="train")
        
        # Category names for cassava leaf disease
        categories = [
            'Cassava Bacterial Blight (CBB)',
            'Cassava Brown Streak Disease (CBSD)',
            'Cassava Green Mottle (CGM)',
            'Cassava Mosaic Disease (CMD)',
            'Healthy'
        ]
        
        print("Successfully loaded Cassava Leaf Disease dataset")
        print(f"Number of images: {len(dataset)}")
        print(f"Number of disease categories: {len(categories)}")
        print("Disease categories:", categories)
        
        return dataset, categories
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have an internet connection")
        print("2. Install the datasets package: pip install datasets")
        print("3. The dataset is about 1.5GB in size")
        print("4. If the dataset fails to load, you can view it here: https://huggingface.co/datasets/cassava-leaf-disease")
        print("5. For a smaller dataset, we can try a different approach")
        raise

# Load the dataset
try:
    dataset, categories = load_pest_dataset()
    
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit(1)

def get_random_example(dataset):
    """Get a random example from the dataset."""
    idx = random.randint(0, len(dataset) - 1)
    example = dataset[idx]
    return example, idx

def print_example_info(example, categories, idx):
    """Print information about an example."""
    print("\n=== Example Information ===")
    print(f"Image index: {idx}")
    
    # Get the image and label
    image = example['image']
    label = example['label']
    
    # Print image information
    if isinstance(image, Image.Image):
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
    elif isinstance(image, dict) and 'bytes' in image:
        try:
            img = Image.open(io.BytesIO(image['bytes']))
            print(f"Image size: {img.size}")
            print(f"Image mode: {img.mode}")
        except Exception as e:
            print(f"Could not process image: {e}")
    
    # Print label information
    if isinstance(label, (int, str)):
        label_idx = int(label)
        if 0 <= label_idx < len(categories):
            print(f"Disease: {categories[label_idx]} (ID: {label_idx})")
        else:
            print(f"Label index out of range: {label_idx}")
    else:
        print(f"Unexpected label format: {label}")

# Print dataset information
print("\n=== Dataset Information ===")
print(dataset)
print(f"\nNumber of examples: {len(dataset)}")

# Get a random example and print its information
try:
    example, idx = get_random_example(dataset)
    print_example_info(example, categories, idx)
except Exception as e:
    print(f"Error getting example: {e}")
    print("This might be due to the dataset structure not matching our expectations.")
    print("Let's try to inspect the dataset structure:")
    if len(dataset) > 0:
        print("\nFirst example keys:", list(dataset[0].keys()))
        if 'label' in dataset[0]:
            print("Label type:", type(dataset[0]['label']))
        if 'image' in dataset[0]:
            print("Image type:", type(dataset[0]['image']))
    else:
        print("The dataset appears to be empty.")

def visualize_example(example, categories):
    """Visualize an example with its label."""
    try:
        # Get the image and label
        image = example['image']
        label = example['label']
        
        # Convert image to PIL Image if needed
        if isinstance(image, dict) and 'bytes' in image:
            img = Image.open(io.BytesIO(image['bytes']))
        elif isinstance(image, Image.Image):
            img = image
        else:
            print("Unsupported image format")
            return None
            
        # Create a copy to draw on
        img_with_text = img.copy()
        draw = ImageDraw.Draw(img_with_text)
        
        # Get the label text
        if isinstance(label, (int, str)):
            label_idx = int(label)
            if 0 <= label_idx < len(categories):
                label_text = f"{categories[label_idx]} (ID: {label_idx})"
            else:
                label_text = f"Unknown label: {label_idx}"
        else:
            label_text = f"Label: {label}"
        
        # Add label text to the image
        draw.text((10, 10), label_text, fill="red")
        
        return img_with_text
        
    except Exception as e:
        print(f"Error visualizing example: {e}")
        return None
        
    except Exception as e:
        print(f"Error visualizing example: {e}")
        return None

# Show random examples
plt.figure(figsize=(15, 10))
for i in range(4):  # Show 4 examples in a 2x2 grid
    plt.subplot(2, 2, i+1)
    try:
        example, _ = get_random_example(dataset)
        img = visualize_example(example, categories)
        if img is not None:
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Example {i+1}")
        else:
            raise Exception("Could not visualize image")
    except Exception as e:
        plt.text(0.5, 0.5, f'Error: {str(e)[:50]}...', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes,
                fontsize=8)
        plt.axis('off')
        print(f"Error showing example {i+1}: {e}")

plt.tight_layout()
plt.show()