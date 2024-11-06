import os
import glob
import re
import random
import numpy as np
import pandas as pd
import cv2
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




def add_gaussian_noise(image, mean=0, var=0.01):
    return random_noise(image, mode='gaussian', mean=mean, var=var)

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    return random_noise(image, mode='s&p', salt_vs_pepper=salt_prob / (salt_prob + pepper_prob), amount=salt_prob + pepper_prob)

def add_speckle_noise(image, mean=0, var=0.01):
    return random_noise(image, mode='speckle', mean=mean, var=var)

def add_poisson_noise(image):
    return random_noise(image, mode='poisson')



def process_and_generate_images(data_path, csv_path, save_path, timestep=7000, test=False):
    """
    Generate and save or display RF images with noise augmentations based on the provided CSV data.

    Parameters:
    - data_path (str): Path to the directory containing RF data folders.
    - csv_path (str): Path to the CSV file containing geometry information.
    - save_path (str): Directory where generated images will be saved.
    - timestep (int): Timestep used for generating RF images.
    - test (bool): If True, displays one example with noise augmentations; otherwise, saves all images.

    Returns:
    - None: The function saves images or displays an example, depending on the `test` parameter.
    """

    folders = sorted([f for f in glob.glob(os.path.join(data_path, '*')) if os.path.isdir(f)])

    # Load target geometries from the CSV file
    geometry_df = pd.read_csv(csv_path)
    target_geometries = geometry_df['shape'].to_numpy()
    pattern_num = r'\d+'

    # Select one random folder and geometry if test mode is enabled
    if test:
        folders = [random.choice(folders)]
        target_geometries = [random.choice(target_geometries)]
    
    for folder in folders:
        skull_name = os.path.basename(folder)
        skull_num = int(re.search(pattern_num, skull_name).group())
        folder_name = f"Syn_{skull_num:03d}"
        new_dir = os.path.join(save_path, folder_name)

        if not test:
            os.makedirs(new_dir, exist_ok=True)
        
        sub_folders = sorted(list_folders(folder))
        for geometry in target_geometries:
            pattern_gp = f"gp_{geometry+1:03d}"
            for s in sub_folders:
                if re.search(pattern_gp, s):
                    csv_file = fetch_csv(s)[0]
                    rf_image = signal_to_image(csv_file, timestep=timestep, pixel=256, figure=False)
                    name = f"Syn_{skull_num:03d}_t_{timestep}_GP_{geometry+1:03d}.png"

                    # Apply noise augmentations
                    noises = {
                        "gaussian": add_gaussian_noise(rf_image),
                        "salt_and_pepper": add_salt_and_pepper_noise(rf_image),
                        "speckle": add_speckle_noise(rf_image),
                        "poisson": add_poisson_noise(rf_image)
                    }

                    if test:
                        # Display the original and noisy images
                        fig, axes = plt.subplots(1, len(noises) + 1, figsize=(15, 5))
                        axes[0].imshow(rf_image, cmap='gray')
                        axes[0].set_title("Original")
                        axes[0].axis('off')
                        for i, (noise_type, noisy_img) in enumerate(noises.items(), start=1):
                            axes[i].imshow(noisy_img, cmap='gray')
                            axes[i].set_title(noise_type.capitalize())
                            axes[i].axis('off')
                        plt.suptitle(f"RF Image with Noise Augmentations (Skull {skull_num}, Geometry {geometry+1})")
                        plt.show()
                        return  # Exit after displaying one example

                    # Save each noisy version if not in test mode
                    for noise_type, noisy_img in noises.items():
                        noisy_img_name = f"{os.path.splitext(name)[0]}_{noise_type}.png"
                        save_image(noisy_img, new_dir, noisy_img_name)





def apply_offset(csv_path, data_path, skull_path, timestep=7000, y_offset=0, random_skull=None):
    """
    Processes RF and skull images, applies an offset along the y-axis to boundaries from the skull overlay, and displays the result.

    Parameters:
    - csv_path (str): Path to the CSV file containing target geometries.
    - data_path (str): Path to the directory containing RF data folders.
    - skull_path (str): Path to the directory containing skull images.
    - y_offset (int): Offset amount along the y-axis for the boundaries drawn.
    - timestep (int): Number of samples or time steps for signal processing.
    - random_skull (int or None): If specified, use this skull index; otherwise, select randomly.
    """
    # List folders in the data path
    folders = sorted([f for f in glob.glob(os.path.join(data_path, '*')) if os.path.isdir(f)])

    # Choose a random skull folder if not specified
    random_skull = random.randint(0, len(folders) - 1) if random_skull is None else random_skull
    skull_num = os.path.basename(folders[random_skull])
    print(f"Picked: {skull_num}")
    
    # List subfolders in the chosen skull folder
    sub_folders = sorted([f for f in glob.glob(os.path.join(folders[random_skull], '*')) if os.path.isdir(f)])
    
    # Load target geometries from the CSV file
    geometry_df = pd.read_csv(csv_path)
    target_geometries = geometry_df['shape'].to_numpy()
    target_csv = []
    for c in target_geometries:
        pattern = f"gp_{c+1:03d}"
        for s in sub_folders:
            if re.search(pattern, s):
                target_csv.append(s)

    # Select a random CSV file from target_csv
    if target_csv:
        random_csv_folder = random.choice(target_csv)
        csv_file = fetch_csv(random_csv_folder)[0]  # Assuming fetch_csv fetches CSV files from a directory
        rf_image_hilbert = signal_to_image(csv_file, pixel=256, figure=False, timestep=timestep)
    else:
        raise ValueError("No matching CSV files found in the target folders.")
    
    # Normalize the rf_image_hilbert
    rf_image_hilbert = cv2.normalize(rf_image_hilbert, None, 0, 255, cv2.NORM_MINMAX)

    # Load and resize skull image to 256x256
    skull_image_path = glob.glob(os.path.join(skull_path, skull_num + ".png"))[0]
    skull_image = cv2.imread(skull_image_path, cv2.IMREAD_GRAYSCALE)
    skull_image_resized = cv2.resize(skull_image, (256, 256))

    # Threshold to extract white area
    _, mask = cv2.threshold(skull_image_resized, 200, 255, cv2.THRESH_BINARY)
    skull_image_rgb = cv2.cvtColor(skull_image_resized, cv2.COLOR_GRAY2BGR)

    # Draw red boundaries on skull image
    red = [255, 0, 0]
    for col in range(mask.shape[1]):
        non_zero_indices = np.nonzero(mask[:, col])[0]
        if non_zero_indices.size > 0:
            top_row = non_zero_indices[0]
            bottom_row = non_zero_indices[-1]
            skull_image_rgb[top_row, col] = red
            skull_image_rgb[bottom_row, col] = red

    # Prepare overlapped image
    overlapped_image = rf_image_hilbert.astype(np.uint8)
    overlapped_image_colored = cv2.cvtColor(overlapped_image, cv2.COLOR_GRAY2BGR)

    # Draw red boundaries from skull image onto the overlapped image with y_offset
    for col in range(skull_image_rgb.shape[1]):
        non_zero_indices = np.nonzero(mask[:, col])[0]
        if non_zero_indices.size > 0:
            top_row = non_zero_indices[0] + y_offset
            bottom_row = non_zero_indices[-1] + y_offset
            overlapped_image_colored[top_row, col] = red
            overlapped_image_colored[bottom_row, col] = red

    # Display the result
    plt.figure(figsize=(6, 6))
    plt.title(f'RF signal at {timestep} with {y_offset} y-axis offset')
    plt.imshow(overlapped_image_colored)
    plt.axis('off')
    plt.show()



def process_and_save_offset_skull_masks(input_path, output_path, y_offset, test=False):
    """
    Processes and saves or displays skull masks with a specified offset.

    Parameters:
    - input_path (str): Directory containing original skull images.
    - output_path (str): Directory to save the processed offset images.
    - y_offset (int): Number of pixels to offset in the y-axis.
    - test (bool): If True, displays one processed example with offset; otherwise, saves all images.

    Returns:
    - None: The function saves offset masks directly to the specified output path or displays an example.
    """
    # Ensure output directory exists only if not in test mode
    if not test:
        os.makedirs(output_path, exist_ok=True)

    # Find all PNG images in the input directory
    skull_images = sorted(glob.glob(os.path.join(input_path, '*.png')))

    # Create padding to add the offset
    pad = np.zeros((y_offset, 256)).astype(np.uint8)

    # Process each skull image
    for s in skull_images:
        skull_image = cv2.imread(s, cv2.IMREAD_GRAYSCALE)
        skull_image_resized = cv2.resize(skull_image, (256, 256))  # Resampling to (256x256)
        cropped_image = skull_image_resized[:-y_offset, :]
        offset_skull = np.concatenate((pad, cropped_image), axis=0)

        # Generate a new filename with offset information
        skull_num = int(re.search(r"\d+", os.path.basename(s)).group())
        name = f"Offset_{y_offset:03d}_Syn_{skull_num:03d}.png"

        if test:
            # Display the processed image with offset as an example
            plt.figure(figsize=(2.56, 2.56), dpi=100)
            plt.imshow(offset_skull, cmap='gray')
            plt.axis('off')
            plt.title(f"Example: Offset {y_offset} pixels (Skull {skull_num})")
            plt.show()
            return  # Exit after displaying one example

        # Save the offset skull image if not in test mode
        fig = plt.figure(figsize=(2.56, 2.56), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(offset_skull, cmap='gray')
        fig.savefig(os.path.join(output_path, name), dpi=100)
        plt.close(fig)

    print(f"Processed and saved {len(skull_images)} images with an offset of {y_offset} pixels.")