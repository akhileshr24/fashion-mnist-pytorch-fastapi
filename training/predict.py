import torch

from explore_data import test_dataset
from model import FashionCNN


class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# Create model
model = FashionCNN()


# Load trained model
model.load_state_dict(
    torch.load(
        "models/model.pth",
        weights_only=True
    )
)

model.eval()


# Get one test image
image, label = test_dataset[0]


# Add batch dimension
image = image.unsqueeze(0)


with torch.no_grad():

    output = model(image)

    # Convert scores to probabilities
    probabilities = torch.softmax(
        output,
        dim=1
    )

    # Get predicted class
    prediction = probabilities.argmax(
        dim=1
    ).item()

    # Get confidence
    confidence = probabilities.max().item()


predicted_class = class_names[prediction]

actual_class = class_names[label]


print(
    "Real answer:",
    actual_class
)

print(
    "Prediction:",
    predicted_class
)

print(
    "Confidence:",
    confidence * 100,
    "%"
)
