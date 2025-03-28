import torch
from torch.utils.data import Dataset
import pandas as pd
import os
# We'll likely use PIL (Pillow) for image loading as it integrates well with torchvision transforms
from PIL import Image # Use Pillow (PIL)

class MuraDataset(Dataset):
    """
    Custom PyTorch Dataset for the MURA dataset.
    Loads image paths and study labels, maps them, and provides
    images and labels for training/validation.
    """
    def __init__(self, csv_image_paths, csv_study_labels, base_data_path, transform=None):
        """
        Args:
            csv_image_paths (string): Path to the csv file with image paths.
            csv_study_labels (string): Path to the csv file with study paths and labels.
            base_data_path (string): Base directory of the MURA dataset (e.g., '.../MURA-v1.1').
                                     Used to construct full image paths if paths in CSV are relative.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__() # Initialize the parent Dataset class

        print(f"Loading dataset: Image paths from '{csv_image_paths}', Study labels from '{csv_study_labels}'")

        # --- Load CSVs ---
        try:
            self.df_img_paths = pd.read_csv(csv_image_paths, header=None, names=['image_path'])
            self.df_study_labels = pd.read_csv(csv_study_labels, header=None, names=['study_path', 'label'])
            print("CSVs loaded successfully in Dataset.")
        except FileNotFoundError as e:
            print(f"!!! ERROR in Dataset init: Could not load CSVs: {e}")
            raise # Re-raise the error to stop execution if files aren't found

        self.base_data_path = base_data_path # Store base path if needed (depends on CSV content)
        self.transform = transform # Store transform function (for preprocessing/augmentation)

        # --- TODO: Map image paths to labels ---
        # We need to create a list or structure (e.g., self.image_label_list)
        # where each entry contains:
        # 1. The full path to an individual image.
        # 2. The corresponding study label (0 or 1).
        # This will likely involve iterating through self.df_img_paths,
        # extracting the study path for each image, and looking up the
        # label in self.df_study_labels.

        # --- Map image paths to labels ---
        print("Mapping image paths to study labels...")
        self.image_label_list = []
        # Create a dictionary for fast lookup of study labels {study_path: label}
        # Ensure study paths from CSV have a trailing slash if needed for matching
        study_label_dict = self.df_study_labels.set_index('study_path')['label'].to_dict()

        missing_labels = 0
        for img_path in self.df_img_paths['image_path']:
            # Extract study path from image path
            # Assumes format like 'MURA-v1.1/train/XR_BODYPART/patient/study/image.png'
            # os.path.dirname gets the directory containing the file
            study_path = os.path.dirname(img_path) + '/' # Add trailing slash

            # Look up the label in our dictionary
            label = study_label_dict.get(study_path) # Use .get() to return None if key not found

            if label is not None:
                # Store the full path (or relative path based on base_data_path) and label
                # Let's store the path exactly as it appears in the image paths CSV for now
                self.image_label_list.append({'image_path': img_path, 'label': label})
            else:
                # This shouldn't happen if CSVs are consistent, but good to check
                missing_labels += 1
                # print(f"Warning: No label found for study path extracted from {img_path}: {study_path}")

        if missing_labels > 0:
            print(f"Warning: Could not find labels for {missing_labels} image paths.")
        else:
            print(f"Successfully mapped {len(self.image_label_list)} images to labels.")

        if not self.image_label_list:
            print("!!! WARNING: image_label_list is empty after mapping! Check paths and logic.")
        # ------------------------------------


    def __len__(self):
        """Returns the total number of images in the dataset."""
        # This should return the length of the structure created in __init__
        return len(self.image_label_list)


    def __getitem__(self, idx):
        """
        Fetches the image and label at the given index.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (image, label) where image is the transformed image tensor
                   and label is the integer label (0 or 1).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # --- TODO: Get image path and label for index 'idx' ---
        # Use the self.image_label_list created in __init__
        
        # --- Get image path and label for index 'idx' ---
        try:
            sample_info = self.image_label_list[idx]
            img_path = sample_info['image_path']
            label = sample_info['label']
        except IndexError:
            print(f"!!! ERROR in __getitem__: Index {idx} out of bounds for dataset length {len(self.image_label_list)}")
            # Decide how to handle this - raise error or return None? Let's raise for now.
            raise IndexError(f"Index {idx} out of bounds.")
        # -------------------------------------------------

        # --- Load Image ---
        # Construct full path if paths in CSV are relative to base_data_path
        # Assuming paths in CSV start like 'MURA-v1.1/...' and base_data_path points to folder *containing* MURA-v1.1
        # full_img_path = os.path.join(self.base_data_path, img_path) # Adjust if needed based on actual paths
        # If paths in CSV are already like '/Users/kyin/.../MURA-v1.1/train/...' then they might be absolute
        # Let's assume paths in CSV are like 'MURA-v1.1/train/...' for now
        # We need to check if base_data_path is the parent of 'MURA-v1.1' or 'MURA-v1.1' itself
        # Assuming base_data_path = '/Users/kyin/Desktop/muraproj/MURA-v1.1'
        # And img_path = 'MURA-v1.1/train/XR_SHOULDER/...'
        # We need to remove the 'MURA-v1.1/' prefix from img_path if base_data_path includes it
        relative_img_path = img_path.replace('MURA-v1.1/', '', 1) # Remove only the first instance
        full_img_path = os.path.join(self.base_data_path, relative_img_path)
        print(f"!!! TODO: Verify full_img_path is correct: {full_img_path} !!!")

        try:
            # Open image using Pillow (convert to RGB in case of grayscale)
            image = Image.open(full_img_path).convert('RGB')
        except FileNotFoundError:
            print(f"!!! ERROR in __getitem__: Image not found at {full_img_path} for index {idx}")
            # Return None or raise error, or return a placeholder? Decide handling.
            # For now, let's return None and handle it in the training loop (or make dummy data)
            # A better approach is to ensure the image_label_list in __init__ is clean.
            return None, None # Or raise error

        # --- Apply transformations ---
        if self.transform:
            image = self.transform(image) # Apply transforms (e.g., resize, tensor conversion, normalization)

        return image, label

# Example usage (for testing later)
if __name__ == '__main__':
    print("Testing MuraDataset...")
    # Replace with actual paths for testing
    base_path = '/Users/kyin/Desktop/muraproj/MURA-v1.1' # Your path
    train_img_csv = os.path.join(base_path, 'train_image_paths.csv')
    train_lbl_csv = os.path.join(base_path, 'train_labeled_studies.csv')

    # We'll define actual transforms later
    dummy_transform = None

    # Create dataset instance (will print TODOs)
    try:
        train_dataset = MuraDataset(
            csv_image_paths=train_img_csv,
            csv_study_labels=train_lbl_csv,
            base_data_path=base_path,
            transform=dummy_transform
        )
        print(f"Dataset created. Length (placeholder): {len(train_dataset)}")

        # TODO: Test __getitem__ once implemented
        # if len(train_dataset) > 0:
        #     img, lbl = train_dataset[0]
        #     if img is not None:
        #         print("Successfully loaded first item.")
        #         # print(f"Image shape/type: {img.shape}, {img.dtype}") # Requires transform to tensor
        #         print(f"Label: {lbl}")
    except Exception as e:
        print(f"Error during dataset testing: {e}")