import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import glob
import re
import random


class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Ensure the image is 2D (grayscale), add channel dimension
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)  # Add channel dimension: [1, H, W]
        
        # Convert image to tensor and normalize
        image_tensor = torch.tensor(image / 255.0, dtype=torch.float32)  # Normalize between [0, 1]
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_tensor, label_tensor



def load_images_by_skull_split(directory, num_train_skulls=80, skull_range=(1, 100), random_split=False):
    """
    Loads images from a directory with subfolders representing different classes and splits them into
    training and validation sets based on skull numbers. It also stores the basenames of the images.
    
    Parameters:
    - directory (str): Path to the main directory containing subfolders (each representing a class).
    - num_train_skulls (int): Number of skulls to be used for the training set (default=80).
    - skull_range (tuple): The range of skull numbers to consider (default=(1, 100)).
    - random_split (bool): Whether to randomly select skulls for training (default=False). If False, it uses the first N skulls in order.
    
    Returns:
    - train_images (list of np.array): Training images.
    - train_labels (list of int): Labels for training images.
    - train_basenames (list of str): Basenames of training images.
    - val_images (list of np.array): Valdiation images.
    - val_labels (list of int): Labels for validation images.
    - val_basenames (list of str): Basenames of validation images.
    - class_names (list of str): List of class names based on subfolder names.
    """
    train_images = []
    train_labels = []
    train_basenames = []
    val_images = [] 
    val_labels = []
    val_basenames = []
    class_names = []
    
    # Regular expression to extract skull number from file names (Syn_{number}_)
    pattern = r'Syn_(\d+)_'
    
    # Get the list of all subfolders (each subfolder represents a class)
    folders = [f for f in glob.glob(os.path.join(directory, '*')) if os.path.isdir(f)]
    print(f"Found {len(folders)} classes.")
    
    # Randomly or sequentially select skull numbers for splitting
    all_skull_numbers = list(range(skull_range[0], skull_range[1] + 1))
    if random_split:
        train_skull_numbers = random.sample(all_skull_numbers, num_train_skulls)
    else:
        train_skull_numbers = all_skull_numbers[:num_train_skulls]
    val_skull_numbers = [skull for skull in all_skull_numbers if skull not in train_skull_numbers]
    
    print(f"Training skull numbers: {sorted(train_skull_numbers)}")
    print(f"Validation skull numbers: {sorted(val_skull_numbers)}")

    pattern_num = r'\d+'
    for folder in folders:
        class_name = int(re.search(pattern_num, os.path.basename(folder)).group())
        class_names.append(class_name)
        print(f"Loading images for class: {class_name}")
        image_paths = glob.glob(os.path.join(folder, '*.png'))  # Adjust for other image formats if needed
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            match = re.match(pattern, filename)
            if match:
                skull_number = int(match.group(1))  # Extract the skull number
                ## Open the image and convert it to grayscale
                img = Image.open(image_path).convert('L')  # 'L' mode converts image to grayscale
                img_array = np.array(img)
                ## Split images based on skull number
                if skull_number in train_skull_numbers:
                    train_images.append(img_array)
                    train_labels.append(class_name)  # Use folder index as the class label
                    train_basenames.append(filename)  # Store the image basename
                elif skull_number in val_skull_numbers:
                    val_images.append(img_array)
                    val_labels.append(class_name)  # Use folder index as the class label
                    val_basenames.append(filename)  # Store the image basename

    print(f"Loaded {len(train_images)} training images and {len(val_images)} validation images.")
    return train_images, train_labels, train_basenames, val_images, val_labels, val_basenames, class_names

