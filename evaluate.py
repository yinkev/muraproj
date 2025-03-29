import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# Import our custom modules
from data_loader import MuraDataset # Assuming it handles patient splitting internally now based on IDs
from model import get_model # Assuming get_model defines the architecture

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained model on the MURA validation split.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the base MURA dataset directory (e.g., /path/to/MURA-v1.1)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the saved model checkpoint (.pth file).')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Size images were resized to during training.')
    parser.add_argument('--batch_size', type=int, default=32, # Can often use larger batch size for evaluation
                        help='Batch size for validation.')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of worker processes for DataLoader.')
    parser.add_argument('--device', type=str, default='auto',
                        help="Device to use ('cpu', 'cuda', 'mps', or 'auto').")
    args = parser.parse_args()
    return args

def evaluate(args):
    print("--- Starting Evaluation ---")
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

    # --- Define Validation Transform ---
    # Should match the validation transform used during training
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    print("Validation transform defined.")

    # --- Create Validation Dataset ---
    # We need the same patient split as used during training for a fair comparison
    print("Recreating validation dataset using patient split logic...")
    train_img_csv = os.path.join(args.data_dir, 'train_image_paths.csv')
    train_lbl_csv = os.path.join(args.data_dir, 'train_labeled_studies.csv')
    val_split_fraction = 0.15 # MUST match the split used in train.py

    try:
        df_all_train_paths = pd.read_csv(train_img_csv, header=None, names=['image_path'])
        def extract_patient_id(path):
            parts = path.split('/')
            for part in parts:
                if part.startswith('patient'): return part
            return None
        all_patient_ids = df_all_train_paths['image_path'].apply(extract_patient_id).unique()
        all_patient_ids = all_patient_ids[all_patient_ids != None]

        np.random.seed(42) # Use the SAME seed as in train.py for consistency
        np.random.shuffle(all_patient_ids)

        split_index = int((1.0 - val_split_fraction) * len(all_patient_ids))
        # train_patient_ids = all_patient_ids[:split_index] # Not needed for eval
        val_patient_ids = all_patient_ids[split_index:]
        print(f"Identified {len(val_patient_ids)} validation patients (using seed 42).")

        val_dataset = MuraDataset(
            csv_image_paths=train_img_csv,
            csv_study_labels=train_lbl_csv,
            base_data_path=args.data_dir,
            transform=val_transform,
            patient_ids_to_include=list(val_patient_ids) # Use only validation patients
        )
        print(f"Validation dataset size: {len(val_dataset)} images")
        if len(val_dataset) == 0:
             print("!!! ERROR: Validation dataset is empty!")
             return

    except FileNotFoundError as e:
        print(f"Error loading CSVs for validation dataset: {e}")
        return
    except Exception as e:
         print(f"Error creating validation dataset: {e}")
         import traceback
         traceback.print_exc()
         return

    # --- Create DataLoader ---
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False, # No shuffling for evaluation
        num_workers=args.num_workers,
        pin_memory=True
    )
    print("Validation dataloader created.")

    # --- Initialize Model ---
    # Create the model structure (MUST match the saved checkpoint structure)
    # We don't need pretrained=True here as we load weights, freeze=False is safest
    model = get_model(num_classes=2, pretrained=False, freeze_layers=False)
    print("Model structure created.")

    # --- Load Checkpoint ---
    if not os.path.exists(args.checkpoint_path):
         print(f"!!! ERROR: Checkpoint file not found at {args.checkpoint_path}")
         return

    print(f"Loading checkpoint: {args.checkpoint_path}")
    try:
        # Load the dictionary, then load the model's state dict
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False) # Load to CPU first, allow non-weights
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully into model.")
        # Optionally print info from checkpoint
        loaded_epoch = checkpoint.get('epoch', 'N/A')
        loaded_metric = checkpoint.get('best_val_metric', 'N/A')
        print(f"  Checkpoint from Epoch: {loaded_epoch}, Best Val Metric (at save time): {loaded_metric}")

    except Exception as e:
        print(f"!!! Error loading checkpoint state dict: {e}")
        import traceback
        traceback.print_exc()
        return

    model.to(device) # Move model to the selected device
    model.eval() # Set model to evaluation mode IMPORTANT!

    # --- Evaluation Loop ---
    print("\n--- Running Evaluation ---")
    all_preds = []
    all_labels = []
    all_scores_positive = []

    val_pbar = tqdm(val_loader, desc="Evaluating")
    with torch.no_grad(): # Disable gradient calculations
        for inputs, labels in val_pbar:
            # Handle potential errors from dataloader (e.g., missing image skipped)
            if inputs is None or labels is None:
                 print("Warning: Skipping a batch due to previous data loading error.")
                 continue
            if isinstance(labels, torch.Tensor) and -1 in labels: # Check for placeholder labels if used
                 print("Warning: Skipping batch with placeholder labels.")
                 continue

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # Get predictions and scores
            probabilities = torch.softmax(outputs, dim=1)
            scores_positive = probabilities[:, 1]
            _, predicted = torch.max(outputs.data, 1)

            # Store results (move to CPU, convert to numpy)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores_positive.extend(scores_positive.cpu().numpy())

    # --- Calculate and Print Metrics ---
    print("\n--- Evaluation Results ---")
    if not all_labels or not all_preds:
         print("No labels or predictions collected, cannot calculate metrics.")
         return

    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    all_scores_positive_np = np.array(all_scores_positive)

    try:
        val_accuracy = accuracy_score(all_labels_np, all_preds_np)
        print(f"Accuracy: {val_accuracy:.4f}")

        # Handle cases where AUC cannot be calculated (e.g., only one class in labels)
        if len(np.unique(all_labels_np)) > 1:
            val_auc = roc_auc_score(all_labels_np, all_scores_positive_np)
            print(f"AUC:      {val_auc:.4f}")
        else:
            print("AUC:      Cannot be calculated (only one class present in labels)")

        print("\nClassification Report:")
        # target_names = ['Normal (0)', 'Abnormal (1)'] # Optional: for clearer report labels
        print(classification_report(all_labels_np, all_preds_np)) # target_names=target_names))

        print("Confusion Matrix:")
        # Rows are True labels, Columns are Predicted labels
        conf_matrix = confusion_matrix(all_labels_np, all_preds_np)
        print(f"[[TN FP]\n [FN TP]]")
        print(f"{conf_matrix}")

    except Exception as e:
        print(f"Error calculating metrics: {e}")

    print("\n--- Evaluation Finished ---")


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)