import torchvision.transforms as transforms
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
        # Construct the full path to the image file.
        # We assume img_path from the CSV looks like 'MURA-v1.1/train/XR_SHOULDER/...'
        # And self.base_data_path points to the directory *containing* 'MURA-v1.1'
        # OR self.base_data_path points *directly* to 'MURA-v1.1'

        # Let's try to be robust: check if img_path starts with the base_path component
        base_folder_name = os.path.basename(self.base_data_path) # e.g., 'MURA-v1.1'
        if img_path.startswith(base_folder_name + '/'):
            # Path in CSV includes the base folder name, remove it before joining
            # e.g. img_path = 'MURA-v1.1/train/...', base_path = '.../MURA-v1.1'
            # relative_path = 'train/...'
            relative_img_path = img_path.split('/', 1)[1] # Split only on the first '/'
            full_img_path = os.path.join(self.base_data_path, relative_img_path)
        elif img_path.startswith('MURA-v1.1/'):
            # Path in CSV includes 'MURA-v1.1/' but maybe base_path is the parent dir
            # e.g. img_path = 'MURA-v1.1/train/...', base_path = '.../muraproj'
            # Check if the 'MURA-v1.1' folder exists within base_path
            potential_base = os.path.join(self.base_data_path, 'MURA-v1.1')
            if os.path.isdir(potential_base):
                # Construct path relative to the parent of base_path? No, relative to base_path
                # Let's assume base_path IS the parent, so join base_path and img_path
                full_img_path = os.path.join(self.base_data_path, img_path) # This seems unlikely based on CSVs
                # Let's reconsider: If base_path is '/Users/.../muraproj' and img_path is 'MURA-v1.1/train/...'
                # Then full_img_path should be '/Users/.../muraproj/MURA-v1.1/train/...'
                # Which is os.path.join(base_path, img_path)
                # This seems less likely given our setup. Let's stick to the first case for now and test.

                # Let's simplify based on our known path:
                # base_data_path = '/Users/kyin/Desktop/muraproj/MURA-v1.1'
                # img_path = 'MURA-v1.1/train/XR_SHOULDER/...'
                # We need '/Users/kyin/Desktop/muraproj/MURA-v1.1/train/XR_SHOULDER/...'
                # So, remove 'MURA-v1.1/' from img_path and join with base_data_path
                relative_img_path = img_path.split('/', 1)[1] # Split only on the first '/'
                full_img_path = os.path.join(self.base_data_path, relative_img_path)

            else:
                # Cannot determine correct path structure
                print(f"!!! WARNING: Ambiguous path structure. img_path: {img_path}, base_data_path: {self.base_data_path}")
                full_img_path = img_path # Fallback, likely incorrect

        else:
            # Path in CSV might be relative to base_data_path directly (e.g., 'train/XR_SHOULDER/...')
            full_img_path = os.path.join(self.base_data_path, img_path)


        # print(f"Trying to load image from: {full_img_path}") # Uncomment for debugging paths

        try:
            # Open image using Pillow (convert to RGB in case of grayscale/RGBA)
            # Grayscale images need 3 channels for standard pre-trained models
            image = Image.open(full_img_path).convert('RGB')
        except FileNotFoundError:
            print(f"!!! ERROR in __getitem__: Image not found at {full_img_path} (derived from img_path: {img_path}) for index {idx}")
            # Raise error to stop DataLoader if an image is missing
            raise FileNotFoundError(f"Image not found: {full_img_path}")
        except Exception as e:
            print(f"!!! ERROR loading image {full_img_path}: {e}")
            # Raise error for other image loading issues
            raise e
        # -----------------------

        # --- Apply transformations ---
        if self.transform:
            image = self.transform(image) # Apply transforms (e.g., resize, tensor conversion, normalization)

        return image, label

# Example usage and testing
if __name__ == '__main__':
    print("\n--- Testing MuraDataset ---")
    # Replace with actual paths for testing
    base_path = '/Users/kyin/Desktop/muraproj/MURA-v1.1' # Your path
    train_img_csv = os.path.join(base_path, 'train_image_paths.csv')
    train_lbl_csv = os.path.join(base_path, 'train_labeled_studies.csv')

    # --- Define Transforms ---
    # Common practice: Resize to input size expected by pre-trained models (e.g., 224x224 or 320x320)
    # Use ImageNet mean and std dev for normalization with transfer learning
    image_size = 224 # Or 320, depending on model choice later
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Define separate transforms for training (with augmentation) and validation (without)
    # For testing the dataset class itself, we only need a basic transform for now
    basic_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), # Resize the image
        transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor (scales to [0, 1])
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std) # Normalize using ImageNet stats
    ])
    print(f"Using basic transform with image size: {image_size}x{image_size}")
    # -------------------------

    # Create dataset instance
    try:
        print("\nCreating dataset instance...")
        train_dataset = MuraDataset(
            csv_image_paths=train_img_csv,
            csv_study_labels=train_lbl_csv,
            base_data_path=base_path,
            transform=basic_transform # Use the defined transform
        )
        print(f"Dataset created successfully. Length: {len(train_dataset)}")

        # --- Test __getitem__ ---
        if len(train_dataset) > 0:
            print("\nAttempting to load first item using __getitem__...")
            # Fetch the first sample (index 0)
            image_tensor, label = train_dataset[0]

            if image_tensor is not None:
                print("Successfully loaded first item (Index 0).")
                print(f"  Image Tensor Shape: {image_tensor.shape}") # Should be [3, image_size, image_size]
                print(f"  Image Tensor Datatype: {image_tensor.dtype}") # Should be torch.float32
                print(f"  Label: {label}")
                # Check tensor value range (should be roughly normalized around 0)
                print(f"  Image Tensor Min value: {image_tensor.min():.4f}")
                print(f"  Image Tensor Max value: {image_tensor.max():.4f}")
                print(f"  Image Tensor Mean value: {image_tensor.mean():.4f}")

            else:
                print("!!! Failed to load first item (returned None). Check __getitem__ logic.")

            # Optional: Test another item
            print("\nAttempting to load another item (Index 100)...")
            image_tensor_100, label_100 = train_dataset[100]
            if image_tensor_100 is not None:
                 print("Successfully loaded item at Index 100.")
                 print(f"  Label: {label_100}")
            else:
                 print("!!! Failed to load item at Index 100.")

        else:
            print("Dataset is empty, cannot test __getitem__.")

    except Exception as e:
        print(f"\n!!! Error during dataset testing: {e}")
        # Print traceback for more details during debugging
        import traceback
        traceback.print_exc()