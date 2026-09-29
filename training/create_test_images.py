from pathlib import Path

from PIL import Image
from torchvision import datasets, transforms

output_folder = Path("test_images")
output_folder.mkdir(exist_ok=True)


class_names = [
    "tshirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle_boot"
]


dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=None
)


found = set()


for image, label in dataset:

    if label not in found:

        image.save(output_folder / f"{class_names[label]}.png")

        found.add(label)

        print(f"Created: {class_names[label]}.png")

    if len(found) == 10:
        break
