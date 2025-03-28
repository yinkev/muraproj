import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2, pretrained=True, freeze_layers=True):
    """
    Loads a pre-trained ResNet-50 model and replaces the final classifier layer.

    Args:
        num_classes (int): Number of output classes (2 for Normal/Abnormal).
        pretrained (bool): Whether to load weights pre-trained on ImageNet.
        freeze_layers (bool): Whether to freeze the parameters of the convolutional layers.
                              Usually True for initial fine-tuning.

    Returns:
        torch.nn.Module: The modified ResNet-50 model.
    """
    print(f"Loading ResNet-50 model (pretrained={pretrained}, freeze_layers={freeze_layers})...")

    # Load the pre-trained ResNet-50 model
    # Use weights=models.ResNet50_Weights.IMAGENET1K_V2 for newer torchvision
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # --- Freeze convolutional layers ---
    if freeze_layers and pretrained:
        print("Freezing convolutional layers...")
        for param in model.parameters():
            # By default, all parameters require gradients. Freeze all initially.
            param.requires_grad = False
        # Note: We will unfreeze the final classifier layer below implicitly
        # by replacing it with a new layer that requires gradients by default.

    # --- Replace the final fully connected layer ---
    # ResNet-50's final layer is named 'fc'
    num_ftrs = model.fc.in_features # Get the number of input features to the original fc layer
    print(f"Original classifier input features: {num_ftrs}")

    # Create a new nn.Linear layer with the correct number of output classes
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(f"Replaced classifier layer. New output classes: {num_classes}")
    # The new layer `model.fc` will have requires_grad=True by default,
    # so its weights will be trained.

    return model

# Example usage and testing
if __name__ == '__main__':
    print("\n--- Testing get_model function ---")

    # Test case 1: Pretrained, Frozen
    model_frozen = get_model(num_classes=2, pretrained=True, freeze_layers=True)
    print("\nModel Structure (Frozen - only fc layer trainable):")
    # Count trainable parameters
    total_params_frozen = sum(p.numel() for p in model_frozen.parameters())
    trainable_params_frozen = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params_frozen:,}")
    print(f"  Trainable parameters: {trainable_params_frozen:,}")
    print("  Final layer (model.fc):", model_frozen.fc)


    # Test case 2: Pretrained, Unfrozen (fine-tuning all layers)
    model_unfrozen = get_model(num_classes=2, pretrained=True, freeze_layers=False)
    print("\nModel Structure (Unfrozen - all layers potentially trainable):")
    total_params_unfrozen = sum(p.numel() for p in model_unfrozen.parameters())
    trainable_params_unfrozen = sum(p.numel() for p in model_unfrozen.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params_unfrozen:,}")
    print(f"  Trainable parameters: {trainable_params_unfrozen:,}") # Should equal total params
    print("  Final layer (model.fc):", model_unfrozen.fc)

    # Test case 3: Not Pretrained (training from scratch - generally not recommended here)
    # model_scratch = get_model(num_classes=2, pretrained=False, freeze_layers=False) # freeze_layers is irrelevant if not pretrained
    # print("\nModel Structure (From Scratch):")
    # trainable_params_scratch = sum(p.numel() for p in model_scratch.parameters() if p.requires_grad)
    # print(f"  Trainable parameters: {trainable_params_scratch:,}")