import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.signal import hilbert
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


# Paths to model files
densenet_path = os.path.abspath("DenseNet/model")
sys.path.append(densenet_path)
from DenseNet import DenseNet

hybrid_path = os.path.abspath("Hybrid_DenseNet_UNet/model")
sys.path.append(hybrid_path)
from Hybrid_DNUN import Hybrid_DenseNet_UNet



def signal_to_image(path, timestep=7000, pixel=256, figure=False):
    """
    This function processes an RF signal from a CSV file and converts it into a grayscale image representation.
    
    Process steps:
        1. Import CSV file containing RF signals using pandas.
        2. Crop the RF signals to a specified number of timesteps.
        3. Apply Hilbert transformation to generate the analytic signal and extract the amplitude envelope.
        4. Perform feature-wise Min-Max scaling of the amplitude envelopes.
        5. Resample the signal using FFT-based interpolation to a target size (e.g., 256x256) for both time and feature dimensions.
        6. Perform an overall Min-Max rescaling on the final resampled signal to normalize values between 0 and 1.
        7. Optionally, plot the raw RF signals and the resulting grayscale image for visualization.
        
    Parameters:
        path (str): The path to the CSV file containing the RF signal data.
        timestep (int): The number of timesteps to crop the signal (default is 7000).
        pixel (int): The target image size for both dimensions (e.g., 256x256).
        figure (bool): If True, visualizes the RF signals and the resulting image (default is False).
    
    Returns:
        np.ndarray: The processed RF signal in the form of a 2D grayscale image.
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




def ArrayReconstruction(array_csv_path, predicted_class):
    """
    This function processes an array CSV file and reconstructs a 2D NumPy array containing 
    the x, y, and z coordinate data for a given predicted class.

    Steps:
    1. Load the CSV file containing medoid and cluster information.
    2. Handle missing values in the 'medoid' column by filling NaNs with 0, and convert the column to integers.
    3. Filter the DataFrame for rows corresponding to the specified predicted class and for rows where 'medoid' equals 1.
    4. Extract columns starting with 'x_', 'y_', and 'z_' to get the x, y, and z coordinates.
    5. Convert the extracted coordinate columns to NumPy arrays.
    6. Concatenate the x, y, and z arrays along the row axis (axis=0) to produce a final array with shape (3, n_features), 
       where 3 represents the x, y, and z coordinates and n_features represents the number of points/features.

    Args:
        array_csv_path (str): The file path to the CSV containing the array data.
        predicted_class (int): The class value to filter the 'cluster' column for the reconstruction.

    Returns:
        np.ndarray: A NumPy array of shape (3, n_features) containing the x, y, and z coordinate data.

    Example:
        coordinates_array = ArrayReconstruction("path/to/file.csv", 2)
    """
    df = pd.read_csv(array_csv_path)
    df['medoid'] = df['medoid'].fillna(0)
    df['medoid'] = df['medoid'].astype(int)
    df_cluster = df[df["cluster"].isin([predicted_class])]
    df_medoid = df_cluster[df_cluster["medoid"].isin([1])]
    
    ## Extract coordinate columns (x_, y_, z_)
    x_coordinates = [col for col in df_medoid.columns if col.startswith('x_')]
    y_coordinates = [col for col in df_medoid.columns if col.startswith('y_')]
    z_coordinates = [col for col in df_medoid.columns if col.startswith('z_')]

    ## Directly convert the selected columns to individual NumPy arrays
    x_array = df_medoid[x_coordinates].to_numpy()
    y_array = df_medoid[y_coordinates].to_numpy()
    z_array = df_medoid[z_coordinates].to_numpy()
    
    ## Concatenate x, y, and z arrays along axis=0 to have (3, 32)
    coordinates_array = np.concatenate([x_array, y_array, z_array], axis=0)

    return coordinates_array 



def inference(rf_image, config, visualization=True):
    """
    Perform inference using pre-trained DenseNet and Hybrid DenseNet-UNet models for ultrasound (US) array estimation 
    and skull segmentation from an RF image.

    The function loads the pre-trained DenseNet model to estimate the US array shape class and the Hybrid DenseNet-UNet 
    model to segment the skull. It converts the RF image into a tensor, applies the models for predictions, reconstructs 
    the 3D array position, and optionally visualizes the results.

    Parameters:
        rf_image (np.ndarray): The input RF image, a 2D grayscale image derived from RF signal data.
        config (Namespace): Configuration object containing model paths and device information.
            - `config.device`: The device for computations (e.g., "cpu", "cuda", or "mps").
            - `config.DenseNet_path`: File path to the pre-trained DenseNet model.
            - `config.Hybrid_DenseNet_UNet_path`: File path to the pre-trained Hybrid DenseNet-UNet model.
            - `config.array_csv_path`: File path to the CSV file containing medoid and cluster information.
            - `config.num_classes` (optional): The number of classes for DenseNet. Defaults to 17 if not set.
        visualization (bool): If True, displays the original RF image and the skull segmentation result.

    Returns:
        tuple: A tuple containing:
            - predicted_array_position (np.ndarray): The reconstructed array positions as a (3, n_features) array, where 
              each row represents x, y, and z coordinates for the array points.
            - predicted_skull_image (np.ndarray): The segmented skull image, a 2D grayscale image.
    """
    

    # Set default number of classes if not specified in config
    if not hasattr(config, 'num_classes'):
        config.num_classes = 17

    ## Convert 'numpy.ndarray' to 'tensor'
    if config.device.type == "mps":  # If using MPS (Apple Metal)
        tensor_image = transforms.ToTensor()(rf_image).to(torch.float32).to(config.device)
        print("!!! Using MPS - Converted input to float32 for compatibility !!!")
    else:
        tensor_image = transforms.ToTensor()(rf_image).to(config.device).double()  # Default for other devices
    
    ## Add a batch dimension
    tensor_image = tensor_image.unsqueeze(0)  # Adds batch dimension to make it 4D
    print(f"Size of 'tensor_image' = {tensor_image.size()}")  # Should print something like [1, channels, height, width]

    ## Load DenseNet
    model_array = DenseNet(num_classes=config.num_classes).to(config.device)
    model_array.load_state_dict(torch.load(config.DenseNet_path, map_location=config.device))
    if config.device.type == "mps":  # If using MPS (Apple Metal)
        model_array = model_array.to(torch.float32)
        print("!!! Model parameters are in float32 instead of float64 for MPS compatibility !!!")
    else:
        model_array = model_array.double()  # Ensure float64 for CUDA
        print("!!! Dense_UNet model parameters are in float64 for CUDA compatibility !!!")
    model_array.eval()  # Set the model to evaluation mode

    
    ## Load Dense_UNet
    model_skull = Hybrid_DenseNet_UNet(model_array).to(config.device)
    # model_skull = DenseNetEncoder_UNetDecoder_v2(model_array).to(config.device)
    model_skull.load_state_dict(torch.load(config.Hybrid_DenseNet_UNet_path, map_location=config.device))

    ## Ensure model parameters are in float32 for MPS or float64 for CUDA
    if config.device.type == "mps":
        model_skull = model_skull.float()  # Ensure float32 for MPS
        print("!!! Dense_UNet model parameters are in float32 for MPS compatibility !!!")
    else:
        model_skull = model_skull.double()  # Ensure float64 for CUDA
        print("!!! Dense_UNet model parameters are in float64 for CUDA compatibility !!!")

    model_skull.eval()  # Set the model to evaluation mode
    
    ## Run inference
    with torch.no_grad():
        output_array = model_array(tensor_image)
        predicted_array = torch.argmax(output_array, dim=1)  # Get class with the highest score
        
        output_skull = model_skull(tensor_image)
        predicted_skull = (torch.sigmoid(output_skull) > 0.5).float()

    ## US array reconstruction
    predicted_class = predicted_array.item()
    print(f"Predicted class = {predicted_class}")
    predicted_array_position = ArrayReconstruction(config.array_csv_path, predicted_class)

    ## 'predicted_skull' to a grayscale image.
    predicted_skull_image = predicted_skull.squeeze().cpu().numpy()

    if visualization:
        plt.figure(figsize=(10, 5))

        ## Original RF Image
        plt.subplot(1, 2, 1)
        plt.imshow(rf_image, cmap='gray')  
        plt.title('RF Image')
        plt.axis('off')

        ## Predicted Skull Segmentation
        plt.subplot(1, 2, 2)
        plt.imshow(predicted_skull_image, cmap='gray')  
        plt.title('Predicted Skull')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    
    return predicted_array_position, predicted_skull_image


