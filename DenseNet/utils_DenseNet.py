import os
import glob
import re
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import hilbert, resample
from PIL import Image
import matplotlib.pyplot as plt
from skimage.util import random_noise


def list_folders(directory):
    """
    List all folders within a specified directory.

    This function scans the provided directory and returns a list of paths for all subdirectories.
    If an error occurs (e.g., if the directory does not exist), it returns the error message as a string.

    Parameters:
    - `directory` (str): The path to the directory to be scanned.

    Returns:
    - `folders` (list of str): A list of paths to subdirectories within the specified directory.
    - If an error occurs, returns a string describing the exception.
    """
    try:
        folders = [f for f in glob.glob(os.path.join(directory, '*')) if os.path.isdir(f)]
        return folders
    except Exception as e:
        return str(e)
    


def fetch_csv(directory):
    """
    List all CSV files within a specified directory.

    This function scans the provided directory and returns a list of paths for all files with a `.csv` extension.
    If an error occurs (e.g., if the directory does not exist), it returns the error message as a string.

    Parameters:
    - `directory` (str): The path to the directory to be scanned.

    Returns:
    - `files` (list of str): A list of paths to CSV files within the specified directory.
    - If an error occurs, returns a string describing the exception.
    """
    try:
        files = [f for f in glob.glob(os.path.join(directory, '*.csv')) if os.path.isfile(f)]
        return files
    except Exception as e:
        return str(e)



def signal_to_image(path, timestep=7000, pixel=256, figure=False):
    """
    Convert an RF signal stored in a CSV file to a rescaled RF image representation.
    
    The function performs the following operations:
    1. Reads the RF signal from a CSV file using pandas.
    2. Crops the signal to a specified timestep length.
    3. Applies Hilbert transformation to extract the analytic signal and computes the amplitude envelope.
    4. Applies feature-wise min-max rescaling for normalization.
    5. Resamples the signal using FFT-based interpolation to a specified pixel size in both time and feature dimensions.
    6. Applies overall min-max rescaling to scale the image between 0 and 1.
    
    Optionally, if `figure=True`, the function plots:
    - The original RF signals for each element.
    - The generated RF image.

    Parameters:
    - `path` (str): Path to the CSV file containing RF signals.
    - `timestep` (int): Number of timesteps to crop the RF signal. Default is 7000.
    - `pixel` (int): The size of the final image in pixels. Default is 256.
    - `figure` (bool): Whether to display the RF signals and image plots. Default is False.

    Returns:
    - `rf_image` (numpy.ndarray): The generated RF image, scaled between 0 and 1.
    """
    raw_rf_signal = pd.read_csv(path)
    rf_signal_array = raw_rf_signal.to_numpy()
    if timestep > rf_signal_array.shape[0]:
        raise ValueError(f"The RF signal collection time was too short! Shorter than {timestep}")
    rf_signal = rf_signal_array[:timestep, :]
    
    transformed_signals = hilbert(rf_signal)
    amplitude_envelope = np.abs(transformed_signals)
    scaler = MinMaxScaler()
    scaled_signal = scaler.fit_transform(amplitude_envelope)
    resized_image = resample(scaled_signal, pixel, axis=0)  # Resample time dimension
    resized_image = resample(resized_image, pixel, axis=1)  # Resample feature dimension
    
    ## rf_image = scaler.fit_transform(resized_image)
    rf_image = (resized_image - np.min(resized_image)) / (np.max(resized_image) - np.min(resized_image))

    if figure:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))

        ## Display RF signal
        s = np.max(rf_signal) * 1
        x = np.arange(timestep)
        for i in range(rf_signal.shape[1]):
            ax1.plot(rf_signal[:, i] + s * i, x, color='black')
    
        labels = ["{}".format(i + 1) for i in range(rf_signal.shape[1])]
        ax1.set_xticks(np.arange(rf_signal.shape[1]) * s)
        ax1.set_xticklabels(labels, fontsize=12)
        for t in ax1.get_xticklabels():
            t.set_color('black')
    
        ax1.set_xlabel("Element Number", fontsize=14)
        ax1.set_ylabel("Time (microseconds)", fontsize=14)
        ax1.set_ylim([len(x), 0])  # Reverse the limits to invert the y-axis so the values descend
        ax1.set_xlim([-s, s * 32])  # Set the x-axis limits
        ax1.set_title(f"RF signals of {rf_signal.shape[1]} elements", fontsize=18)
    
        ## Display RF image
        ax2.imshow(rf_image, cmap='gray')
        ax2.set_title(f'RF image {rf_image.shape}', fontsize=18)
        ax2.axis('off')  # Optional: Remove axis for a cleaner look
    
        plt.tight_layout()
        plt.show()

    return rf_image



def augment_image_with_varied_noise(image):
    """
    Apply random noise to an input image for data augmentation purposes.
    
    The function randomly selects one of four noise types and applies it with varying parameters:
    1. **Gaussian Noise**: Adds Gaussian-distributed noise with a random variance between 0.005 and 0.015.
    2. **Salt and Pepper Noise**: Randomly switches pixels to black or white, with a random `amount` between 0.03 and 0.06 and a `salt_vs_pepper` ratio between 0.3 and 0.7.
    3. **Speckle Noise**: Adds noise proportional to the pixel value, with a random variance between 0.005 and 0.015.
    4. **Poisson Noise**: Applies noise based on Poisson distribution (no configurable parameters).

    Parameters:
    - `image` (numpy.ndarray): The input image to which noise will be added.

    Returns:
    - `augmented_image` (numpy.ndarray): The image with added noise for augmentation.
    """
    noise_type = random.choice(['gaussian', 's&p', 'speckle', 'poisson'])
    
    if noise_type == 'gaussian': # Vary the variance for Gaussian noise
        var = random.uniform(0.005, 0.015)  # Random variance between 0.005 and 0.02
        return random_noise(image, mode='gaussian', var=var)
    
    elif noise_type == 's&p': # Vary the amount and salt_vs_pepper ratio for Salt and Pepper noise
        amount = random.uniform(0.03, 0.06)  # Random amount between 0.02 and 0.1
        salt_vs_pepper = random.uniform(0.3, 0.7)  # Random ratio between 0.3 and 0.7
        return random_noise(image, mode='s&p', amount=amount, salt_vs_pepper=salt_vs_pepper)
    
    elif noise_type == 'speckle': # Vary the variance for Speckle noise
        var = random.uniform(0.005, 0.015)  # Random variance between 0.005 and 0.02
        return random_noise(image, mode='speckle', var=var)
    
    elif noise_type == 'poisson': # Poisson noise has no configurable parameters, so just apply it
        return random_noise(image, mode='poisson')



def save_image(image, save_path, img_name):
    """
    Saves an image as a PNG file.
    Parameters:
    - `image` (numpy.ndarray): The image to save.
    - `save_path` (str): The directory to save the image.
    - `img_name` (str): The filename for the saved image.
    """
    fig = plt.figure(figsize=(2.56, 2.56), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, cmap='gray')
    fig.savefig(os.path.join(save_path, img_name), dpi=100)
    plt.close(fig)




def augment_and_save_images(original_images_all, save_path, timesteps, aug_required):
    """
    Apply noise augmentation to existing RF images and save the augmented versions.
    
    This function takes a dictionary of original images categorized by timesteps and applies 
    random noise augmentations to generate more training data. The augmented images are saved 
    in the corresponding class directories.

    Steps:
    1. Iterate through the timesteps for each original image.
    2. Randomly select an image and apply one of four noise types:
       - Gaussian Noise
       - Salt and Pepper Noise
       - Speckle Noise
       - Poisson Noise
    3. Save the augmented images with a unique filename that indicates augmentation.

    Parameters:
    - `original_images_all` (dict): A nested dictionary with original RF images categorized by timesteps and keys.
    - `save_path` (str): The directory where augmented images will be saved.
    - `timesteps` (numpy.ndarray): An array of timesteps used for categorizing augmentation.
    - `aug_required` (int): The total number of augmentations required for each class.

    Returns:
    - None: The function saves augmented images directly to the specified save path.
    """
    for t in timesteps:
        for augment_num in range(aug_required // len(timesteps)):
            selected_key = random.choice(list(original_images_all[t].keys()))
            selected_image = original_images_all[t][selected_key]
            augmented_image = augment_image_with_varied_noise(selected_image)
            img_name = f"aug_t_{t:05d}_GP_{selected_key:03d}_aug{augment_num}.png"
            save_image(augmented_image, save_path, img_name)




def process_and_generate_images(df, path, save_path, timesteps, cluster_counts_dict):
    """
    Process CSV data to generate and save RF images with augmentations for each class.
    
    This function iterates through directories containing RF data, matches them with 
    the corresponding cluster in the DataFrame, and generates RF images based on specific
    timesteps. It also saves these images and calls another function to apply noise 
    augmentation if required.

    Steps:
    1. List all folders in the specified path and match them with shapes in the DataFrame.
    2. Generate RF images from CSV files by reading them and converting them to image representations.
    3. Save generated images to class-specific directories.
    4. Prepare a structure for augmenting images using varied noise types.

    Parameters:
    - `df` (pandas.DataFrame): The DataFrame containing cluster and shape information for matching.
    - `path` (str): The directory path containing subdirectories with RF data.
    - `save_path` (str): The directory where generated images will be saved.
    - `timesteps` (numpy.ndarray): An array of timesteps used for generating RF images.
    - `cluster_counts_dict` (dict): A dictionary with cluster counts, indicating the number of images per class.

    Returns:
    - None: The function saves images directly to the specified save path.
    """
    pattern_num = r'\d+'
    max_images_per_class = max(cluster_counts_dict.values())
    
    folders = sorted(list_folders(path))
    print("Number of folders (Synthetic skulls) =", len(folders))
    for c in cluster_counts_dict.keys():
        class_dir = os.path.join(save_path, f"class_{c}")
        os.makedirs(class_dir, exist_ok=True)
        
        aug_required = len(timesteps) * (max_images_per_class - cluster_counts_dict[c])
        print(f"Class: {c} - {aug_required} augmentations required")
        
        shapes = np.array(df[df['cluster'] == c]['shape'].tolist()) + 1
        original_images_all = {t: {} for t in timesteps}
        
        for folder in folders:
            skull_num = int(re.search(pattern_num, os.path.basename(folder)).group())
            sub_folders = sorted(list_folders(folder))

            for t in timesteps:
                for sub_folder in sub_folders:
                    gp_num = int(re.search(pattern_num, os.path.basename(sub_folder)).group())
                    if gp_num in shapes:
                        csv_files = fetch_csv(sub_folder)
                        if csv_files:
                            rf_image = signal_to_image(csv_files[0], timestep=t, pixel=256, figure=False)
                            img_name = f"Syn_{skull_num:03d}_t_{t:05d}_GP_{gp_num:03d}.png"
                            save_image(rf_image, class_dir, img_name)
                            original_images_all[t][gp_num] = rf_image

        augment_and_save_images(original_images_all, class_dir, timesteps, aug_required)

        num_of_images = len(glob.glob(os.path.join(class_dir, '*')))
        print(f"{num_of_images} images in class {c}")





