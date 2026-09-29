import torch

from src.explore_data import test_dataset
from src.model import FashionCNN

# Class names
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
    torch.load("model.pth")
)


# Evaluation mode
model.eval()


# Get one image from test dataset
image, label = test_dataset[0]


# Add batch dimension
image = image.unsqueeze(0)


# Make prediction
with torch.no_grad():

    output = model(image)

    probabilities = torch.softmax(output, dim=1)

    prediction = probabilities.argmax(dim=1).item()

    confidence = probabilities.max().item()

# Get class name
predicted_class = class_names[prediction]

actual_class = class_names[label]


# Print results
print("Real answer:", actual_class)
print("Prediction:", predicted_class)
print("Confidence:", confidence * 100, "%")
