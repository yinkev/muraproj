# MURA Musculoskeletal X-ray Abnormality Detection

## Overview

This project applies deep learning techniques, specifically transfer learning with a ResNet-50 model using PyTorch, to classify upper extremity radiographs (X-rays) from the MURA dataset as either 'Normal' (containing no abnormalities) or 'Abnormal' (containing at least one abnormality). This was undertaken as an independent project during OMS1 to explore the application of computer science skills to medical imaging analysis relevant to Orthopedics.

**Final Model Performance (on Validation Set):**
*   AUC: ~0.860
*   Accuracy: ~81.0%
*   (See Results section for more details)

## Project Goal

*   To build an end-to-end pipeline for medical image classification using Python and PyTorch.
*   To gain practical experience with data loading, preprocessing, data augmentation, transfer learning, model training, evaluation metrics (AUC, Accuracy, Precision, Recall), and checkpointing.
*   To apply these techniques to a relevant orthopedic imaging dataset (MURA).
*   To create a well-documented project suitable for a portfolio.

## Dataset

*   **Source:** MURA (musculoskeletal radiographs) dataset from Stanford ML Group. [Link to dataset website: https://stanfordmlgroup.github.io/competitions/mura/]
*   **Content:** ~40,000 upper extremity X-ray images from ~12,000 studies, covering Finger, Hand, Wrist, Elbow, Forearm, Humerus, Shoulder.
*   **Labels:** Provided at the *study* level (Normal/Abnormal), not per image.
*   **Access:** Requires signing a Data Use Agreement.
*   **Split:** A patient-level split (85% train / 15% validation) was created from the official training set to ensure model evaluation on unseen patients (using random seed 42 for reproducibility). The official validation set labels were not used for validation metric calculation during training.

## Setup

```bash
**1. Clone Repository:**
git clone https://github.com/yinkev/muraproj.git # Use your actual repo URL
cd muraproj

**2. Create Virtual Environment (using venv):**
python3 -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# or .\.venv\Scripts\activate  # On Windows

**3. Install Dependencies:**
pip install --upgrade pip
pip install -r requirements.txt

**4. Download Data:**
Download the MURA dataset from the official source (requires DUA).
Extract the MURA-v1.1 folder.
Important: Update the --data_dir argument in the train.py and evaluate.py scripts (or pass it via command line) to point to the full path of the extracted MURA-v1.1 folder.
```

## Usage
1. Training:
The script uses command-line arguments for configuration. Key arguments:
--data_dir: Path to MURA-v1.1 folder (Required).
--epochs: Number of epochs to train for.
--batch_size: Batch size.
--lr: Learning rate.
--freeze: Add this flag to only train the final classifier layer. Omit to fine-tune all layers.
--load_checkpoint: Path to a .pth checkpoint file to resume training from.
--output_dir: Folder to save checkpoints (defaults to outputs/).

Example (Initial Frozen Training):
```bash
python train.py --data_dir /path/to/MURA-v1.1 --epochs 20 --batch_size 16 --freeze
```

Example (Fine-tuning from checkpoint):
```bash
python train.py --data_dir /path/to/MURA-v1.1 --load_checkpoint outputs/best_model_checkpoint.pth --epochs 30 --lr 1e-5
```
The best model checkpoint (best_model_checkpoint.pth) is saved in the specified --output_dir based on validation AUC.

2. Evaluation:
Evaluates a saved model checkpoint on the validation set.
--data_dir: Path to MURA-v1.1 folder (Required).
--checkpoint_path: Path to the saved .pth model checkpoint (Required).

Example:
```bash
python evaluate.py --data_dir /path/to/MURA-v1.1 --checkpoint_path outputs/best_model_checkpoint.pth
```
Outputs Accuracy, AUC, Classification Report, and Confusion Matrix.

## Methodology
**Model:**
ResNet-50 architecture pre-trained on ImageNet, loaded via torchvision. The final fully connected layer was replaced with a new one outputting 2 classes (Normal/Abnormal).

**Training Strategy:**
Initial Phase: Trained only the final classifier layer (--freeze flag) for ~15-20 epochs with a learning rate of 1e-4.
Fine-tuning Phase: Loaded the best checkpoint from the initial phase, unfroze all layers, and continued training for ~10-15 more epochs with a lower learning rate (1e-5).

**Data Handling:**
MuraDataset class implemented to load images and map them to study-level labels.
Patient-level 85/15 Train/Validation split created from the official training set.
**Preprocessing:** Images resized to 224x224, converted to PyTorch tensors, and normalized using ImageNet statistics.
**Augmentation (Training Only):** Random horizontal flips and random rotations (up to 10 degrees) were applied.
**Loss Function:** Cross-Entropy Loss (nn.CrossEntropyLoss).
**Optimizer:** AdamW.
**Scheduler:** Learning rate reduced on plateau (ReduceLROnPlateau) monitoring validation AUC during the fine-tuning phase.
**Evaluation Metric:** Model checkpointing based on highest validation AUC.

## Results
The best model was achieved after Epoch 27 (during the fine-tuning phase) with the following performance on the validation set:
AUC: 0.8597
Accuracy: 0.8101

**Classification Report:**
```bash
              precision    recall  f1-score   support

           0       0.81      0.88      0.85      3278
           1       0.81      0.71      0.75      2314

    accuracy                           0.81      5592
   macro avg       0.81      0.79      0.80      5592
weighted avg       0.81      0.81      0.81      5592
```

**Confusion Matrix:**
```bash
[[TN FP]  [[2896  382]
 [FN TP]]  [ 680 1634]]
```
**Interpretation:** The model shows good overall discriminative ability (AUC ~0.86). It performs well on identifying Normal cases (Recall 0.88). Its main weakness is missing ~29% of Abnormal cases (Recall 0.71), resulting in 680 False Negatives on this validation split. When it predicts Abnormal, it is correct ~81% of the time (Precision 0.81).

## Limitations & Future Work
**Binary Classification Only:** Model distinguishes Normal vs. Abnormal, but not the type of abnormality.

**Dataset Labels:** Relies on study-level labels; image-level labels could potentially improve performance. Label noise/subjectivity might exist.

**Validation Set:** Validation performed on a split of the original training data. Performance on the official MURA validation set (labels not publicly released post-competition) or external datasets is unknown.

**Recall for Abnormalities:** While improved during fine-tuning, recall for abnormal cases (sensitivity) could potentially be further improved (e.g., weighted loss, different architectures, more augmentation).

**Computational Limits:** Training performed primarily on Apple Silicon MPS, limiting experimentation with very large models or extensive hyperparameter searches compared to high-end NVIDIA GPUs.

**Future Work:**
Experiment with different architectures (EfficientNet, DenseNet).
Implement more advanced data augmentations.
Try weighted loss functions to improve abnormal recall.
Systematic hyperparameter tuning.
(If possible) Evaluate on external datasets.

## Dependencies
See requirements.txt.

Key libraries include:
PyTorch & Torchvision
Pandas
NumPy
OpenCV-Python
Scikit-learn
tqdm
tabulate (for explore_data.py)