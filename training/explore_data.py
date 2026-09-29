
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()


train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
# image, label = train_dataset[0]

# print("Type:", type(image))
# print("Shape:", image.shape)
# print("Dtype:", image.dtype)
# print("Min:", image.min())
# print("Max:", image.max())

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

images, labels = next(iter(train_loader))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
