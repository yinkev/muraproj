import pandas as pd
import os

# --- IMPORTANT: REPLACE WITH YOUR ACTUAL PATH TO THE MURA FOLDER ---
# Example: mura_data_base_path = '/Users/kyin/Downloads/MURA-v1.1'
# Example: mura_data_base_path = '/Volumes/ExternalDrive/MURA-v1.1'
mura_data_base_path = '/Users/kyin/Desktop/muraproj/MURA-v1.1'
# -------------------------------------------------------------------

print(f"Attempting to load data from: {mura_data_base_path}")

# Construct paths to the CSV files
train_img_paths_csv = os.path.join(mura_data_base_path, 'train_image_paths.csv')
train_labels_csv = os.path.join(mura_data_base_path, 'train_labeled_studies.csv')
valid_img_paths_csv = os.path.join(mura_data_base_path, 'valid_image_paths.csv')

# Load the CSVs into Pandas DataFrames
try:
    df_train_paths = pd.read_csv(train_img_paths_csv, header=None, names=['image_path'])
    df_train_labels = pd.read_csv(train_labels_csv, header=None, names=['study_path', 'label'])
    df_valid_paths = pd.read_csv(valid_img_paths_csv, header=None, names=['image_path'])
    print("CSVs loaded successfully!")
except FileNotFoundError as e:
    print(f"\n!!! ERROR loading CSV: {e}")
    print(f"!!! Please ensure mura_data_base_path is set correctly.")
    print(f"!!! The path currently points to: '{mura_data_base_path}'")
    exit() # Exit if files not found

# Display basic info and first few rows
print("\n--- Training Image Paths (First 5) ---")
print(df_train_paths.head().to_markdown(index=False)) # Use markdown for better formatting

print("\n--- Training Study Labels (First 5) ---")
print(df_train_labels.head().to_markdown(index=False))

print("\n--- Label distribution (Train) ---")
# Calculate and print percentages
label_counts = df_train_labels['label'].value_counts()
label_percentages = df_train_labels['label'].value_counts(normalize=True) * 100
print(f"Label 0 (Normal):   {label_counts.get(0, 0)} ({label_percentages.get(0, 0):.2f}%)")
print(f"Label 1 (Abnormal): {label_counts.get(1, 0)} ({label_percentages.get(1, 0):.2f}%)")


print("\n--- Validation Image Paths (First 5) ---")
print(df_valid_paths.head().to_markdown(index=False))

# --- How to link image path to study label? ---
print("\n--- Example: Linking Image Path to Study Label ---")
if not df_train_paths.empty:
    example_img_path = df_train_paths['image_path'].iloc[0] # Get the first image path
    print(f"Example Image Path: {example_img_path}")

    # Extract study path from image path (assuming structure like 'MURA-v1.1/train/XR_BODYPART/patient/study/')
    # Method 1: Using os.path.dirname multiple times
    study_path_from_image = os.path.dirname(example_img_path) + '/' # Add trailing slash for matching
    print(f"Extracted Study Path (Method 1): {study_path_from_image}")

    # Method 2: Splitting the string (less robust but illustrates)
    parts = example_img_path.split('/')
    if len(parts) >= 5:
         # Join the parts up to the study folder name
         study_path_from_image_split = "/".join(parts[:-1]) + "/"
         print(f"Extracted Study Path (Method 2): {study_path_from_image_split}")

    # Find the label for this study path in the labels DataFrame
    # Note: This requires the study paths in df_train_labels to match exactly how we extract them
    matching_label_row = df_train_labels[df_train_labels['study_path'] == study_path_from_image] # Use Method 1 extraction

    if not matching_label_row.empty:
        label = matching_label_row['label'].iloc[0]
        print(f"Found Label for Study Path: {label}")
    else:
        print(f"!!! Could not find label for extracted study path: {study_path_from_image}")
        print("!!! Check if paths in train_labeled_studies.csv have trailing slashes.")