import os
import glob
import re
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, rfimage_folder_path, skull_image_path, num_train_skulls=80, skull_range=(1, 100), random_split=False, transform=None, mode='train'):
        """
        Initializes the ImageDataset class with paths, parameters, and settings.
        
        Args:
            rfimage_folder_path (str): Path to the folder containing 'Syn_{number}' folders with RF images.
            skull_image_path (str): Path to the folder containing target skull images (e.g., Synthetic_{number}.png).
            num_train_skulls (int): Number of skulls to be used for training (default=80).
            skull_range (tuple): Range of skull numbers (default=(1, 100)).
            random_split (bool): Whether to randomly select skulls for training (default=False).
            transform (callable, optional): Optional transform function to be applied on an image.
            mode (str): Mode of the dataset, either 'train' or 'val' for validation (default='train').
        """
        self.rfimage_folder_path = rfimage_folder_path
        self.skull_image_path = skull_image_path
        self.transform = transform or transforms.ToTensor()  # Ensure a default transformation is applied
        self.mode = mode

        self.pattern = r"\d+"  # Regular expression to extract numbers from folder names
        self.rfimage_folders = sorted(glob.glob(os.path.join(self.rfimage_folder_path, '*')))  # Load all RF image folders
        self.skull_images = sorted(glob.glob(os.path.join(self.skull_image_path, '*.png')))  # Load all skull images

        ## Debug: Print the number of skull images and RF image folders
        # print(f"Found {len(self.skull_images)} skull images.")
        # print(f"Found {len(self.rfimage_folders)} folders matching 'Syn_*' pattern.")

        ## Randomly or sequentially select skull numbers for splitting
        all_skull_numbers = list(range(skull_range[0], skull_range[1] + 1))
        if random_split:
            self.train_skull_numbers = random.sample(all_skull_numbers, num_train_skulls)
        else:
            self.train_skull_numbers = all_skull_numbers[:num_train_skulls]
        self.val_skull_numbers = [skull for skull in all_skull_numbers if skull not in self.train_skull_numbers]

        ## Print based on mode
        if self.mode == 'train':
            print(f"Training skull numbers: {sorted(self.train_skull_numbers)}")
        elif self.mode == 'val':
            print(f"Validation skull numbers: {sorted(self.val_skull_numbers)}")

    
    def __len__(self):
        """
        Returns the total number of images in the dataset for the specified mode (train or val).
        
        Returns:
            int: Total number of images in the dataset.
        """
        total_images = 0
        for folder in self.rfimage_folders:
            folder_name = os.path.basename(folder)
            skull_num = int(re.search(self.pattern, folder_name).group())
            if self.mode == 'train' and skull_num in self.train_skull_numbers:
                total_images += len(glob.glob(os.path.join(folder, "*.png")))
            elif self.mode == 'val' and skull_num in self.val_skull_numbers:
                total_images += len(glob.glob(os.path.join(folder, "*.png")))
        return total_images

    
    def __getitem__(self, idx):
        folder_idx = 0
        skull_num = None
    
        ## Loop to find the right folder and adjust idx accordingly
        for folder in self.rfimage_folders:
            folder_name = os.path.basename(folder)
            skull_num = int(re.search(self.pattern, folder_name).group())
            if (self.mode == 'train' and skull_num in self.train_skull_numbers) or (self.mode == 'val' and skull_num in self.val_skull_numbers):
                images_in_folder = sorted(glob.glob(os.path.join(folder, "*.png")))
                if idx < len(images_in_folder):
                    rfimage_path = images_in_folder[idx]
                    break
                idx -= len(images_in_folder)
        

        ## Fetch the corresponding skull image based on the skull number
        skull_image_path = glob.glob(os.path.join(self.skull_image_path, f"*{skull_num:03d}.png"))[0]  # We expect one matching image
    
        ## Load the RF image and skull image
        rf_image = Image.open(rfimage_path).convert('L')  # Load RF image as grayscale
        skull_image = Image.open(skull_image_path).convert('L')  # Load skull image as grayscale
    
        ## Apply the provided transformation (or convert to tensor)
        rf_image = self.transform(rf_image)
        skull_image = self.transform(skull_image)
    
        ## Return RF and skull images along with their basenames and skull number
        rf_basename = os.path.basename(rfimage_path)
        skull_basename = os.path.basename(skull_image_path)
    
        return rf_image, skull_image, skull_num, rf_basename, skull_basename



## Debugging purpose
def show_example(dataset, num_examples=10):
    """
    Randomly selects and displays the basenames of RF and skull images from the dataset without showing figures.
    
    Args:
        dataset (Dataset): The dataset to sample from (train_dataset or val_dataset).
        num_examples (int): Number of examples to display.
    """
    print(f"Displaying {num_examples} random examples from the dataset:")
    indices = random.sample(range(len(dataset)), num_examples) # Randomly select 5 unique indices from the dataset
    for i, idx in enumerate(indices):
        rf_image, skull_image, skull_num, rf_basename, skull_basename = dataset[idx]  # Fetching the examples by index
        print(f"\tExample {i + 1} - RF Image : {rf_basename} & Skull Image: {skull_basename}")
    print('\n')


