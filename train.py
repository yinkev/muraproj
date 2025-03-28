
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import argparse # For command-line arguments
from tqdm import tqdm # For progress bars

# Import our custom modules
from data_loader import MuraDataset
from model import get_model

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on the MURA dataset.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the base MURA dataset directory (e.g., /path/to/MURA-v1.1)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save checkpoints and logs.')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Size to resize images to (e.g., 224, 320).')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and validation.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, # Often start lower for fine-tuning
                        help='Learning rate.')
    parser.add_argument('--num_workers', type=int, default=4, # Adjust based on your system
                        help='Number of worker processes for DataLoader.')
    parser.add_argument('--freeze', action='store_true', # Default is False (fine-tune classifier only)
                        help='Freeze convolutional layers (only train classifier).')
    parser.add_argument('--device', type=str, default='auto',
                        help="Device to use ('cpu', 'cuda', 'mps', or 'auto').")
    # Add more arguments as needed (e.g., model choice, optimizer choice)
    args = parser.parse_args()
    return args

# --- Main Training Function ---
def main(args):
    print("--- Starting Training ---")
    print(f"Arguments: {args}")

    # --- Setup Device ---
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA (NVIDIA GPU)")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(args.device)
        print(f"Using specified device: {device}")

    # --- Create Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # --- Define Transforms ---
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Define Train Transform (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        # --- Add Augmentations ---
        transforms.RandomHorizontalFlip(p=0.5), # Randomly flip images horizontally 50% of the time
        transforms.RandomRotation(degrees=10), # Randomly rotate images by up to 10 degrees
        # transforms.ColorJitter(brightness=0.1, contrast=0.1), # Optional: Adjust brightness/contrast slightly
        # -------------------------
        transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std) # Normalize
    ])
    print("Training transform defined with augmentation (Random HFlip, Random Rotation).") # Ensure this print statement is updated too

    # Define Validation Transform (no augmentation) - This remains the same
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    print("Validation transform defined (no augmentation).")

    # --- Create Datasets with Train/Validation Split ---
    print("Creating datasets with patient-level train/validation split...")
    train_img_csv = os.path.join(args.data_dir, 'train_image_paths.csv')
    train_lbl_csv = os.path.join(args.data_dir, 'train_labeled_studies.csv')
    # valid_img_csv = os.path.join(args.data_dir, 'valid_image_paths.csv') # We won't use the official valid set for validation metrics initially

    # --- Perform Patient-Based Split ---
    val_split_fraction = 0.15 # Use 15% of patients for validation (adjust as needed)
    print(f"Using {val_split_fraction*100:.1f}% of patients for validation.")

    try:
        # Load all training image paths to extract patient IDs
        df_all_train_paths = pd.read_csv(train_img_csv, header=None, names=['image_path'])

        # Extract unique patient IDs
        def extract_patient_id(path):
            parts = path.split('/')
            for part in parts:
                if part.startswith('patient'):
                    return part
            return None

        all_patient_ids = df_all_train_paths['image_path'].apply(extract_patient_id).unique()
        # Remove potential None values if extraction failed for some paths
        all_patient_ids = all_patient_ids[all_patient_ids != None]
        print(f"Found {len(all_patient_ids)} unique patient IDs in the training set.")

        # Shuffle the patient IDs
        np.random.seed(42) # Set seed for reproducibility
        np.random.shuffle(all_patient_ids)

        # Split IDs
        split_index = int((1.0 - val_split_fraction) * len(all_patient_ids))
        train_patient_ids = all_patient_ids[:split_index]
        val_patient_ids = all_patient_ids[split_index:]

        print(f"Splitting into {len(train_patient_ids)} train patients and {len(val_patient_ids)} validation patients.")

    except FileNotFoundError as e:
        print(f"Error loading train_image_paths.csv for splitting: {e}")
        return # Exit if we can't load the paths for splitting

    # --- Create Actual Datasets using the patient ID lists ---
    try:
        print("Initializing Training Dataset...")
        train_dataset = MuraDataset(
            csv_image_paths=train_img_csv,
            csv_study_labels=train_lbl_csv,
            base_data_path=args.data_dir,
            transform=train_transform,
            patient_ids_to_include=list(train_patient_ids) # Pass the list of train patient IDs
        )

        print("Initializing Validation Dataset...")
        val_dataset = MuraDataset(
            csv_image_paths=train_img_csv, # Still load paths from the main train CSV
            csv_study_labels=train_lbl_csv, # Still load labels from the main train CSV
            base_data_path=args.data_dir,
            transform=val_transform, # Use validation transform (no augmentation)
            patient_ids_to_include=list(val_patient_ids) # Pass the list of validation patient IDs
        )
        print(f"Actual Train dataset size: {len(train_dataset)} images")
        print(f"Actual Validation dataset size: {len(val_dataset)} images")

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("!!! ERROR: One of the datasets is empty after splitting. Check split logic.")
            return # Exit if split failed

    except Exception as e:
        print(f"Error creating datasets after split: {e}")
        # Print traceback for more details during debugging
        import traceback
        traceback.print_exc()
        return # Exit if datasets fail
    # --------------------------------------------------------

    # --- Create DataLoaders ---
    print("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True, # Shuffle training data
        num_workers=args.num_workers,
        pin_memory=True # Helps speed up data transfer to GPU if using CUDA
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False, # No need to shuffle validation data
        num_workers=args.num_workers,
        pin_memory=True
    )
    print("Dataloaders created.")

    # --- Initialize Model ---
    print("Initializing model...")
    model = get_model(
        num_classes=2,
        pretrained=True,
        freeze_layers=args.freeze # Control freezing via command line
    )
    model.to(device) # Move model to the selected device (GPU/CPU)
    print("Model initialized and moved to device.")

    # --- Define Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    print("Loss function: CrossEntropyLoss")

    # Define optimizer - only pass trainable parameters
    # If freezing, only optimize the classifier head (model.fc)
    # If not freezing, optimize all parameters
    if args.freeze:
         print("Optimizing only the parameters of the final classifier layer.")
         optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr)
    else:
         print("Optimizing all model parameters.")
         optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Optimizer: AdamW, Learning Rate: {args.lr}")

    # --- Training Loop ---
    print("\n--- Starting Training Loop ---")
    best_val_metric = -1.0 # Placeholder for tracking best model (e.g., use AUC or accuracy)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # --- Training Phase ---
        model.train() # Set model to training mode
        running_loss_train = 0.0
        # TODO: Add tracking for training accuracy/metrics if desired

        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        for inputs, labels in train_pbar:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss_train += loss.item() * inputs.size(0)
            train_pbar.set_postfix({'loss': loss.item()}) # Show current batch loss

        epoch_loss_train = running_loss_train / len(train_dataset)
        print(f"Training Loss: {epoch_loss_train:.4f}")

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        running_loss_val = 0.0
        all_preds = []
        all_labels = []
        all_scores_positive = []

        val_pbar = tqdm(val_loader, desc=f"Validate Epoch {epoch+1}")
        with torch.no_grad(): # No need to track gradients during validation
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss_val += loss.item() * inputs.size(0)
                val_pbar.set_postfix({'loss': loss.item()})

                # Store predictions and labels for metrics calculation later
                # Get predicted class index (0 or 1)
                _, predicted = torch.max(outputs.data, 1)

                 # Apply Softmax to get probabilities (optional but good practice for interpretation)
                probabilities = torch.softmax(outputs, dim=1)
                # Get the probability of the positive class (class 1)
                scores_positive = probabilities[:, 1] # Assuming class 1 is the positive class

                # Store scores along with predictions and labels
                all_scores_positive.extend(scores_positive.cpu().numpy()) # Collect positive class scores
                all_preds.extend(predicted.cpu().numpy()) # Move to CPU before converting to numpy
                all_labels.extend(labels.cpu().numpy())

        epoch_loss_val = running_loss_val / len(val_dataset)
        print(f"Validation Loss: {epoch_loss_val:.4f}")

        # --- Calculate validation metrics ---
        # Ensure tensors are on CPU and converted to numpy for sklearn
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)

        # Calculate Accuracy
        val_accuracy = accuracy_score(all_labels_np, all_preds_np)
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Calculate AUC requires scores for the positive class (class 1)
        # We need to modify the loop above to collect scores
        # Let's assume we collected them in a list `all_scores_positive`
        # Placeholder for now - we will modify the loop next
        val_auc = 0.0 # Placeholder
        if 'all_scores_positive' in locals() and len(all_scores_positive) == len(all_labels_np):
             try:
                 all_scores_positive_np = np.array(all_scores_positive)
                 val_auc = roc_auc_score(all_labels_np, all_scores_positive_np)
                 print(f"Validation AUC: {val_auc:.4f}")
             except ValueError as e:
                  print(f"Could not calculate AUC: {e}") # Handle cases with only one class present
        else:
             print("!!! AUC calculation skipped: Positive class scores not collected yet.")


        # Confusion Matrix (Optional but helpful)
        # conf_matrix = confusion_matrix(all_labels_np, all_preds_np)
        # print(f"Confusion Matrix:\n{conf_matrix}")

        # Use AUC as the metric to track for saving the best model
        current_val_metric = val_auc # Or use val_accuracy if AUC fails initially
        # ------------------------------------

        # --- Save Checkpoint ---
        # TODO: Save checkpoint periodically and save the best model based on validation metric
        print("!!! TODO: Implement checkpoint saving (best model based on val_metric) !!!")
        # Example:
        # if current_val_metric > best_val_metric:
        #     best_val_metric = current_val_metric
        #     save_path = os.path.join(args.output_dir, 'best_model.pth')
        #     torch.save(model.state_dict(), save_path)
        #     print(f"Saved new best model to {save_path}")


    print("\n--- Training Finished ---")

# --- Entry Point ---
if __name__ == '__main__':
    args = parse_args()
    main(args)