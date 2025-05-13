# Data Processing Script for Chest X-ray Dataset

import os
import shutil
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def create_directory_structure():
    """Create the directory structure for the project."""
    directories = [
        'data/processed/train/NORMAL',
        'data/processed/train/PNEUMONIA',
        'data/processed/val/NORMAL',
        'data/processed/val/PNEUMONIA',
        'data/processed/test/NORMAL',
        'data/processed/test/PNEUMONIA'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess an image for the model.
    - Resize to target dimensions
    - Convert to RGB (in case of grayscale)
    - Normalize pixel values
    """
    try:
        img = Image.open(image_path)
        # Convert grayscale to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Resize image
        img = img.resize(target_size)
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def process_dataset(raw_data_path, processed_data_path, target_size=(224, 224)):
    """Process all images in the dataset."""
    # Get a list of categories (NORMAL, PNEUMONIA)
    categories = ['NORMAL', 'PNEUMONIA']
    
    # Process each category
    for category in categories:
        # Get list of image files
        source_dir = os.path.join(raw_data_path, category)
        if not os.path.exists(source_dir):
            print(f"Directory not found: {source_dir}")
            continue
            
        image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing {len(image_files)} images for category: {category}")
        
        # Process each image
        for img_file in image_files:
            source_path = os.path.join(source_dir, img_file)
            target_path = os.path.join(processed_data_path, category, img_file)
            
            # Preprocess and save the image
            processed_img = preprocess_image(source_path, target_size)
            if processed_img is not None:
                processed_img = (processed_img * 255).astype(np.uint8)
                Image.fromarray(processed_img).save(target_path)

def split_data(raw_data_path, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split the dataset into train, validation, and test sets.
    Uses the raw data to create symbolic links in the processed folders.
    """
    # Get a list of categories (NORMAL, PNEUMONIA)
    categories = ['NORMAL', 'PNEUMONIA']
    
    for category in categories:
        # Get list of image files
        category_path = os.path.join(raw_data_path, category)
        if not os.path.exists(category_path):
            print(f"Directory not found: {category_path}")
            continue
            
        image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split into train and temporary test set
        train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=random_state)
        
        # Split the train set into train and validation
        train_files, val_files = train_test_split(train_files, test_size=val_size, random_state=random_state)
        
        print(f"Category: {category}")
        print(f"  - Train: {len(train_files)} images")
        print(f"  - Validation: {len(val_files)} images")
        print(f"  - Test: {len(test_files)} images")
        
        # Copy files to respective directories
        for file_list, split_name in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
            target_dir = f"data/processed/{split_name}/{category}"
            for file_name in file_list:
                source_path = os.path.join(raw_data_path, category, file_name)
                target_path = os.path.join(target_dir, file_name)
                shutil.copy(source_path, target_path)

def visualize_samples(processed_data_path, num_samples=5):
    """Visualize some sample images from each category."""
    categories = ['NORMAL', 'PNEUMONIA']
    
    plt.figure(figsize=(12, 6))
    for i, category in enumerate(categories):
        category_path = os.path.join(processed_data_path, 'train', category)
        image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Select random samples
        samples = random.sample(image_files, min(num_samples, len(image_files)))
        
        for j, sample in enumerate(samples):
            img_path = os.path.join(category_path, sample)
            img = Image.open(img_path)
            
            plt.subplot(2, num_samples, i*num_samples + j + 1)
            plt.imshow(img)
            plt.title(category)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('data/sample_images.png')
    plt.close()
    print("Sample visualization saved to data/sample_images.png")

def calculate_dataset_stats(processed_data_path):
    """Calculate and print dataset statistics."""
    categories = ['NORMAL', 'PNEUMONIA']
    splits = ['train', 'val', 'test']
    
    stats = {}
    
    for split in splits:
        stats[split] = {}
        for category in categories:
            category_path = os.path.join(processed_data_path, split, category)
            if os.path.exists(category_path):
                image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                stats[split][category] = len(image_files)
            else:
                stats[split][category] = 0
    
    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 40)
    for split in splits:
        total = sum(stats[split].values())
        print(f"{split.upper()} set: {total} images")
        for category in categories:
            count = stats[split][category]
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  - {category}: {count} images ({percentage:.1f}%)")
    print("-" * 40)

def main():
    # Path definitions
    raw_data_path = "data/raw/chest_xray"  # Change according to your dataset structure
    processed_data_path = "data/processed"
    
    # Create directory structure
    create_directory_structure()
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_raw_path = f"data/raw/chest_xray/{split}"
        split_processed_path = f"data/processed/{split}"
        
        if os.path.exists(split_raw_path):
            process_dataset(split_raw_path, split_processed_path)
        else:
            print(f"Warning: {split_raw_path} directory not found")
    
    # Calculate and print dataset statistics
    calculate_dataset_stats(processed_data_path)
    
    # Visualize sample images
    visualize_samples(processed_data_path)
    
    print("Data preprocessing completed!")

if __name__ == "__main__":
    main()
