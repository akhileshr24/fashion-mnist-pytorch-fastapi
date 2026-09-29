import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Convert image to Tensor
transform = transforms.ToTensor()


# Training dataset
train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# Testing dataset
test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


print("Training samples:", len(train_dataset))
print("Test samples:", len(test_dataset))


# Get one image and label
image, label = train_dataset[0]

print("Image type:", type(image))
print("Label:", label)


# Display image
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()


# Check image information
print("Type:", type(image))
print("Shape:", image.shape)
print("Dtype:", image.dtype)
print("Min:", image.min())
print("Max:", image.max())


# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# Get one batch
images, labels = next(iter(train_loader))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
