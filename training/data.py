import matplotlib.pyplot as plt

from torchvision import datasets, transforms


# Convert images into PyTorch tensors
transform = transforms.ToTensor()


# Load training dataset
train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# Show dataset information
print("Training samples:", len(train_dataset))


# Get one image and label
image, label = train_dataset[0]

print("Image type:", type(image))
print("Label:", label)
print("Shape:", image.shape)
print("Dtype:", image.dtype)
print("Min:", image.min())
print("Max:", image.max())


# Display the image
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()
