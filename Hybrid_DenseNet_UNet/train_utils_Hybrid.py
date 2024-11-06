import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms

densenet_path = os.path.abspath("../DenseNet/model")
sys.path.append(densenet_path)
from DenseNet import DenseNet

from model.Hybrid_DNUN import Hybrid_DenseNet_UNet



## Load Pre-trained DenseNet
def load_pretrained_densenet(config):
    """
    Loads a pre-trained DenseNet model using the configuration provided in the Args class instance.

    Args:
        config (Args): Configuration object containing necessary parameters such as path, device, etc.

    Returns:
        torch.nn.Module: The DenseNet model loaded onto the specified device.
    """
    densenet = DenseNet(num_classes=config.num_classes, dropout_prob=config.dropout_prob) 
    densenet.load_state_dict(torch.load(config.pre_trained_densenet, map_location=config.device))  # Load state dict and map to device
    densenet.to(config.device)
    
    return densenet



## Dice Coefficient Implementation
def dice_coefficient(preds, targets, threshold=0.5):
    """
    Computes the Dice coefficient, a metric commonly used to measure overlap between predicted and target segmentation masks.

    Args:
        preds (torch.Tensor): The predicted segmentation mask from the model.
        targets (torch.Tensor): The ground truth segmentation mask.
        threshold (float): The threshold to binarize predictions, with values above threshold set to 1, others to 0 (default=0.5).

    Returns:
        float: Dice coefficient value, indicating similarity between predicted and target masks (1 means perfect overlap).
    """
    preds = (preds > threshold).float()  # Binarize predictions
    smooth = 1.0  # Smoothing factor to prevent division by zero
    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()



def initialize_weights(model):
    """
    Initializes weights of the model layers using suitable initialization techniques for each layer type.

    Args:
        model (torch.nn.Module): The neural network model whose weights need to be initialized.

    Returns:
        None: The model's weights are modified in place.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Initialize Conv2d with Kaiming normal
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Set Conv2d biases to zero
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)  # Set BatchNorm weights to 1
            nn.init.constant_(m.bias, 0)    # Set BatchNorm biases to zero
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)  # Initialize Linear layers with Kaiming normal
            nn.init.constant_(m.bias, 0)       # Set Linear biases to zero

        

##
def Hybrid_Dense_UNet_training(trained_densenet, config, train_dataset, val_dataset):
    """
    Hybrid_DenseNet_UNet - No bottleneck layer
    Dense UNet Training Loop with Early Stopping and Learning Rate Adjustment Based on Dice Score.
    This function trains a U-Net model using a pre-trained DenseNet encoder. It implements early stopping
    and reduces the learning rate based on the validation Dice score for binary segmentation tasks.
    
    Args:
        trained_densenet (nn.Module): Pre-trained DenseNet model used as the encoder for the UNet.
        config (object): Configuration object with parameters such as batch size, learning rate, etc.
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
    
    Returns:
        None: Trained model is saved to disk along with performance plots and metrics.
    """

    fold_results = {}

    ## Convert the device to a string to make it JSON serializable
    config_dict = config.__dict__.copy()
    config_dict['device'] = str(config.device)
    fold_results['configuration'] = config_dict

    ## Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=config.pin_memory, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=config.pin_memory, num_workers=config.num_workers)

    ## Initialize the model with pre-trained DenseNet as the encoder
    model = Hybrid_DenseNet_UNet(trained_densenet, out_channels=1, output_size=config.output_size).to(config.device)

    ## Freeze the pre-trained DenseNet encoder parameters so they are not updated
    for param in model.encoder.parameters():
        param.requires_grad = False  # Freeze the encoder

    ## Initialize the weights for each decoder block individually
    initialize_weights(model.dec4)
    initialize_weights(model.dec3)
    initialize_weights(model.dec2)
    initialize_weights(model.dec1)
    initialize_weights(model.final_conv)
    
    criterion = nn.BCEWithLogitsLoss()  # Binary classification 

    ## Only update trainable params (decoder)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, weight_decay=config.weight_decay)  
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config.lr_scheduler)

    ## Mixed Precision Scaler for performance optimization (only use for CUDA)
    use_amp = config.device == "cuda"
    if use_amp:
        scaler = GradScaler()

    ## Early stopping and learning rate tracking based on Dice score
    best_train_dice = 0.0  # Track the best train Dice score
    best_val_dice = 0.0  # Track the best validation Dice score
    best_train_accuracy = 0.0  # Track the best train accuracy
    best_val_accuracy = 0.0  # Track the best validation accuracy
    early_stopping_counter = 0
    lr_scheduler_steps = []

    ## Lists to store accuracies, losses, and Dice coefficients
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []

    ## Training loop
    total_start_time = time.time()
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        dice_sum_train = 0.0  # To accumulate the Dice coefficient for training
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} - Training", leave=False)
        for rf_image, skull_image, *_ in train_loader_tqdm:
            rf_image, skull_image = rf_image.to(config.device), skull_image.to(config.device)
            optimizer.zero_grad()

            ## Mixed precision forward pass if using CUDA, else normal precision
            if use_amp:
                with autocast():
                    outputs = model(rf_image)
                    loss = criterion(outputs, skull_image)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(rf_image)
                loss = criterion(outputs, skull_image)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            ## Calculate Dice coefficient and accuracy
            dice_train = dice_coefficient(outputs, skull_image)  # Calculate Dice coefficient
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (predicted == skull_image).sum().item()
            total_train += skull_image.numel()
            dice_sum_train += dice_train

            ## Update the progress bar
            train_loader_tqdm.set_postfix(loss=loss.item(), dice=dice_train)

        ## Calculate average training loss, Dice coefficient, and accuracy for this epoch
        train_loss = running_loss / len(train_loader)
        avg_dice_train = dice_sum_train / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_dices.append(avg_dice_train)
        train_accuracies.append(train_accuracy)

        ## Validation loop
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        dice_sum_val = 0.0
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} - Validation", leave=False)
        with torch.no_grad():
            for rf_image, skull_image, *_ in val_loader_tqdm:
                rf_image, skull_image = rf_image.to(config.device), skull_image.to(config.device)
    
                ## Mixed precision for validation if using CUDA
                if use_amp:
                    with autocast():
                        outputs = model(rf_image)
                        loss = criterion(outputs, skull_image)
                else:
                    outputs = model(rf_image)
                    loss = criterion(outputs, skull_image)
                    
                running_val_loss += loss.item()

                ## Calculate Dice coefficient and accuracy for validation
                dice_val = dice_coefficient(outputs, skull_image)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (predicted == skull_image).sum().item()
                total_val += skull_image.numel()

                dice_sum_val += dice_val

                ## Update the progress bar for validation
                val_loader_tqdm.set_postfix(loss=loss.item(), dice=dice_val)

        ## Calculate average validation loss, Dice coefficient, and accuracy for this epoch
        val_loss = running_val_loss / len(val_loader)
        avg_dice_val = dice_sum_val / len(val_loader) # Epoch-wise dice score
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_dices.append(avg_dice_val)
        val_accuracies.append(val_accuracy)

        ## Reduce learning rate based on validation loss
        # scheduler.step(val_loss)
        scheduler.step(avg_dice_val) # Focusing on the metric that best captures the quality of the segmentation task

        print(f'Epoch [{epoch + 1}/{config.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Dice: {avg_dice_train:.4f}, Val Dice: {avg_dice_val:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')

        ## Check for epoch-wise Dice coefficient
        if avg_dice_train > 0.90 and avg_dice_val > 0.90:
            print("Both training and validation Dice coefficients for the epoch exceeded 90%. Saving model.")
            model_save_path = os.path.join(config.model_save_path, f"Dense_UNet_epoch_{epoch + 1}_train_dice_{avg_dice_train:.4f}_val_dice_{avg_dice_val:.4f}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved with train Dice: {avg_dice_train:.4f} and val Dice: {avg_dice_val:.4f} at epoch {epoch + 1}")

        ## Check for best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            ## Save best model
            if best_val_accuracy > 0.90:
                torch.save(model.state_dict(), os.path.join(config.model_save_path, f"Best_valAcc_Dense_UNet.pth"))
                print(f"Best model saved with validation accuracy: {best_val_accuracy:.4f}")

        ## Track best train accuracy
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy


        ## Track best train dice score
        if avg_dice_train > best_train_dice:
            best_train_dice = avg_dice_train

        ## Check for best epoch-wise validation Dice coefficient (for early stopping)
        if avg_dice_val > best_val_dice:
            best_val_dice = avg_dice_val
            early_stopping_counter = 0  # Reset counter if validation Dice improves
            ## Save best model
            if avg_dice_val > 0.90:
                torch.save(model.state_dict(), os.path.join(config.model_save_path, f"Best_valDice_Dense_UNet.pth"))
                print(f"Best model saved with validation Dice: {avg_dice_val:.4f}")
        else:
            early_stopping_counter += 1
        
        ## Check for early stopping
        if early_stopping_counter >= config.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    ## End of training
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"Total training time: {total_elapsed_time:.2f} seconds")

    ## Log final performance
    print(f"Best Train Dice Score: {best_train_dice:.4f}")
    print(f"Best Validation Dice Score: {best_val_dice:.4f}")

    ## Save results
    fold_results["train_val_split"] = {
        'best_train_accuracy': best_train_accuracy,
        'best_val_accuracy': best_val_accuracy,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_losses': train_losses,  # Save training losses for analysis
        'val_losses': val_losses,  # Save validation losses
        'best_train_dice': best_train_dice,
        'best_val_dice': best_val_dice,
        'train_dices': train_dices,  # Save training Dice scores
        'val_dices': val_dices,  # Save validation Dice scores
        'total_time': total_elapsed_time,
        'lr_scheduler_steps': lr_scheduler_steps  # Save when learning rate decreases
    }

    ## Save the final trained model
    torch.save(model.state_dict(), os.path.join(config.model_save_path, f"Dense_UNet.pth"))
    print(f"Final model saved to Dense_UNet.pth")

    ## Plot train and validation losses, accuracies, and Dice coefficients
    epochs_completed = len(train_losses)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    ## Plot for Train and Validation Losses
    axs[0].plot(range(1, epochs_completed + 1), train_losses, label='Train Loss', color='blue')
    axs[0].plot(range(1, epochs_completed + 1), val_losses, label='Validation Loss', color='orange')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Train and Validation Loss')
    axs[0].legend()

    ## Plot for Train and Validation Accuracies
    axs[1].plot(range(1, epochs_completed + 1), train_accuracies, label='Train Accuracy', color='blue')
    axs[1].plot(range(1, epochs_completed + 1), val_accuracies, label='Validation Accuracy', color='orange')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Train and Validation Accuracy')
    axs[1].legend()

    ## Plot for Train and Validation Dice Coefficients
    axs[2].plot(range(1, epochs_completed + 1), train_dices, label='Train Dice', color='blue')
    axs[2].plot(range(1, epochs_completed + 1), val_dices, label='Validation Dice', color='orange')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Dice Coefficient')
    axs[2].set_title('Train and Validation Dice Coefficient')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config.model_save_path, "train_val_metrics_plot.png"))
    plt.show()

    ## Save all results to a JSON file
    json_file_path = os.path.join(config.model_save_path, "Dense_UNet_results.json")
    with open(json_file_path, "w") as json_file:
        json.dump(fold_results, json_file)

    print(f"Results saved to {json_file_path}")


