import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import json
import time
import os

from models.DenseNet import DenseNet          


def initialize_weights(model):
    """
    Initialize the weights of the model layers for improved training stability and performance.

    This function applies specific initialization techniques based on the type of layer:
    - `nn.Conv2d`: Kaiming normal initialization (suitable for ReLU activations) to improve convergence.
    - `nn.BatchNorm2d`: Initializes weights to 1 and biases to 0, providing neutral initialization for batch normalization.
    - `nn.Linear`: Applies Kaiming normal initialization to weights and sets biases to 0, aiding in stable training.

    Parameters:
    - model (torch.nn.Module): The model whose weights are to be initialized.
    
    Returns:
    - None: The function modifies the model weights in-place.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Initialize convolutional layers with Kaiming normal (He initialization)
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize biases to zero if they exist
        elif isinstance(m, nn.BatchNorm2d):
            # Initialize batch norm weights to 1 and biases to zero
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Initialize fully connected layer weights with Kaiming normal and biases to zero
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)



def DenseNet_training(config, dataset_train, dataset_val):
    """
    Train a DenseNet model using the provided training and validation datasets.
    
    The function performs training with early stopping, learning rate scheduling, and model checkpointing.
    It also logs training progress, validation metrics, and final results, including plots and JSON results.

    Parameters:
    - `config` (Args): A configuration object containing training hyperparameters and settings.
    - `dataset_train` (Dataset): The PyTorch dataset for training data.
    - `dataset_val` (Dataset): The PyTorch dataset for validation data.

    Returns:
    - None: The function trains the model and saves it along with results and plots to the specified directory.
    """
    fold_results = {}
    
    ## Convert the device to a string to make it JSON serializable
    config_dict = config.__dict__.copy()
    config_dict['device'] = str(config.device)
    fold_results['configuration'] = config_dict

    ## Create data loaders for training and validation
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, pin_memory=config.pin_memory, num_workers=config.num_workers)
    val_loader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False,  pin_memory=config.pin_memory, num_workers=config.num_workers)

    ## Initialize the model, apply weight initialization, criterion, and optimizer
    model = DenseNet(num_classes=config.num_classes, dropout_prob=config.dropout_prob).to(config.device)
    initialize_weights(model)  # Apply weight initialization

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config.lr_scheduler)

    ## Mixed Precision Scaler for performance optimization (only use for CUDA)
    use_amp = config.device == "cuda"
    if use_amp:
        scaler = GradScaler()

    ## Early stopping and learning rate tracking
    early_stopping_counter = 0
    best_train_accuracy = 0.0
    best_val_accuracy = 0.0
    early_stopping_epoch = None
    lr_scheduler_steps = []

    ## Lists to store accuracies and losses for plotting
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []  

    ## Training loop
    total_start_time = time.time()
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} - Training", leave=False)  # Show training progress
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(config.device), labels.to(config.device).long()
            optimizer.zero_grad()

            ## Mixed precision forward pass if using CUDA, else normal precision
            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        ## Calculate training accuracy for this epoch
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)
        train_losses.append(running_loss / len(train_loader))

        ## Update the progress bar less frequently
        train_loader_tqdm.set_postfix(loss=running_loss / len(train_loader), accuracy=train_accuracy)

        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy

        ## Validation loop
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} - Validation", leave=False)  # Show validation progress
        with torch.no_grad():
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(config.device), labels.to(config.device).long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # Validation loss
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)
        val_losses.append(running_val_loss / len(val_loader))  # Track validation loss

        ## Update the progress bar less frequently
        val_loader_tqdm.set_postfix(loss=running_val_loss / len(val_loader), accuracy=val_accuracy)

        ## Calculate precision, recall, and F1-score
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=1)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1)
        print(f'Epoch [{epoch + 1}/{config.num_epochs}], Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

        ## Check if both training and validation accuracy are above 90%, then Save the model
        if train_accuracy > 0.90 and val_accuracy > 0.90:
            print("Both training and validation accuracies exceeded 90%. Saving model.")
            model_save_path = os.path.join(config.model_save_path, f"DenseNet_epoch_{epoch + 1}_train_{train_accuracy:.4f}_val_{val_accuracy:.4f}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved with train accuracy: {train_accuracy:.4f} and val accuracy: {val_accuracy:.4f} at epoch {epoch + 1}")
            
            scheduler.step(val_accuracy)  # Apply the scheduler's logic here
            if optimizer.param_groups[0]['lr'] < config.learning_rate:  # Check if the learning rate decreased
                lr_scheduler_steps.append(epoch + 1)

        ## Check for best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            early_stopping_counter = 0  # Reset counter if validation improves
            ## Save best model
            print(f"Model with the best val accuracy {val_accuracy:.4f} was saved at eopch {epoch + 1}")
            torch.save(model.state_dict(), os.path.join(config.model_save_path, f"Best_val_DenseNet.pth"))
        else:
            early_stopping_counter += 1

        ## Early stopping condition
        if early_stopping_counter >= config.patience:
            early_stopping_epoch = epoch + 1
            print(f"Early stopping at epoch {epoch + 1}")
            break

    ## End of training
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"Total training time: {total_elapsed_time:.2f} seconds")

    ## Log final performance
    print(f"Best Train Accuracy: {best_train_accuracy:.4f}")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

    ## Save results
    fold_results["80_20_split"] = {
        'best_train_accuracy': best_train_accuracy,
        'best_val_accuracy': best_val_accuracy,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_losses': train_losses,  # Save training losses for analysis
        'val_losses': val_losses,  # Save validation losses
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_time': total_elapsed_time,
        'early_stopping_epoch': early_stopping_epoch,  # Save when early stopping happens
        'lr_scheduler_steps': lr_scheduler_steps  # Save when learning rate decreases
    }

    ## Save the trained model
    torch.save(model.state_dict(), os.path.join(config.model_save_path, f"DenseNet.pth"))
    print(f"Model saved to DenseNet.pth")

    ## Plot train and validation accuracies and losses
    epochs_completed = len(train_accuracies)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  
    ## Plot for Train and Validation Accuracy
    axs[0].plot(range(1, epochs_completed + 1), train_accuracies, label='Train Accuracy')
    axs[0].plot(range(1, epochs_completed + 1), val_accuracies, label='Validation Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Train and Validation Accuracy')
    axs[0].legend()
    
    ## Plot for Train and Validation Loss
    axs[1].plot(range(1, epochs_completed + 1), train_losses, label='Train Loss')
    axs[1].plot(range(1, epochs_completed + 1), val_losses, label='Validation Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Train and Validation Loss')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.model_save_path,"train_val_accuracy_loss_plot.png"))
    plt.show()

    ## Save all results to a JSON file
    json_file_path = os.path.join(config.model_save_path,f"DenseNet_results.json")
    with open(json_file_path, "w") as json_file:
        json.dump(fold_results, json_file)

    print(f"Training result was saved to {json_file_path}")