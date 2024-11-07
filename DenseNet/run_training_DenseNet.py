import os
import torch
import argparse
import json  

from dataset_DenseNet import ImageDataset, load_images_by_skull_split
from train_utils_DenseNet import DenseNet_training


## Ensure compatibility with MPS (Metal Performance Shaders)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Args:
    def __init__(self, num_classes, model_save_path, best_model_save_path, num_epochs=50, batch_size=128, learning_rate=0.001, weight_decay=1e-4, dropout_prob=0.0, lr_scheduler=None):
        self.num_classes = num_classes
        self.folds = 5  # k-fold
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_prob = dropout_prob
        self.lr_scheduler = lr_scheduler if lr_scheduler else {'mode': 'max', 'factor': 0.5, 'patience': 3, 'verbose': True}
        self.patience = 5  # Early stop
        self.device = device
        self.num_workers = 0
        self.pin_memory = True
        self.model_save_path = model_save_path
        self.best_model_save_path = best_model_save_path

def main():
    parser = argparse.ArgumentParser(description="Train DenseNet model.")
    parser.add_argument('--directory', type=str, required=True, help="Directory containing training images.")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs for training.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for training.")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay for optimizer.")
    parser.add_argument('--dropout_prob', type=float, default=0.0, help="Dropout probability for the model.")
    parser.add_argument('--lr_scheduler', type=str, default='{"mode": "max", "factor": 0.5, "patience": 3, "verbose": true}', help="Learning rate scheduler as a JSON string.")
    parser.add_argument('--model_save_path', type=str, required=True, help="Path to save the model checkpoints.")
    parser.add_argument('--best_model_save_path', type=str, required=True, help="Path to save the best model checkpoint.")
    
    args = parser.parse_args()
    
    # Parse the lr_scheduler argument from JSON string to a dictionary
    lr_scheduler = json.loads(args.lr_scheduler)

    # Load training and validation images
    print("Loading training and validation images...")
    train_images, train_labels, train_basenames, val_images, val_labels, val_basenames, class_names = load_images_by_skull_split(args.directory, random_split=False)
    print(f"Training: Loaded {len(train_images)} images from {len(class_names)} classes")
    print(f"Validation: Loaded {len(val_images)} images from {len(class_names)} classes.")
    
    # Create datasets
    dataset_train = ImageDataset(train_images, train_labels)
    dataset_val = ImageDataset(val_images, val_labels)

    # Create an Args object with the provided command-line arguments
    training_args = Args(
        num_classes=len(class_names),
        model_save_path=args.model_save_path,
        best_model_save_path=args.best_model_save_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_prob=args.dropout_prob,
        lr_scheduler=lr_scheduler
    )

    # Train the model
    DenseNet_training(training_args, dataset_train, dataset_val)

if __name__ == "__main__":
    main()



