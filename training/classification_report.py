import torch
from explore_data import test_dataset
from model import FashionCNN
from sklearn.metrics import classification_report

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


# Load trained weights
model.load_state_dict(
    torch.load(
        "models/model.pth",
        weights_only=True
    )
)


# Evaluation mode
model.eval()


# Store predictions and actual labels
predictions = []
actual = []


# Disable gradients
with torch.no_grad():

    for image, label in test_dataset:

        # Add batch dimension
        image = image.unsqueeze(0)

        # Make prediction
        output = model(image)

        # Get predicted class
        prediction = output.argmax(
            dim=1
        ).item()

        # Store results
        predictions.append(prediction)
        actual.append(label)


# Generate classification report
print(
    classification_report(
        actual,
        predictions,
        target_names=class_names
    )
)
