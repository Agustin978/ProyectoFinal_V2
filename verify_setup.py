
import sys
import os
import torch
try:
    from src.models.densenet import get_model
    from src.data.dataset import NIHChestXRayDataset
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_model():
    print("Testing model instantiation...")
    model = get_model(num_classes=14)
    print("Model instantiated successfully.")
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Model forward pass shape: {y.shape}")
    assert y.shape == (2, 14)
    print("Model shape check passed.")

if __name__ == "__main__":
    test_model()
