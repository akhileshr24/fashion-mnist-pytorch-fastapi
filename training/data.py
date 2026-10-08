import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Convert images into PyTorch tensors
transform = transforms.ToTensor()


# Load the complete training dataset
full_train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# Load the test dataset
test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# Split the training dataset
# 54,000 → training
# 6,000  → validation
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [54000, 6000],
    generator=torch.Generator().manual_seed(42)
)


# Training DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# Validation DataLoader
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)


# Test DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))
