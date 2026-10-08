import torch

from data import test_loader
from model import FashionCNN


# Create model
model = FashionCNN()


# Load the best trained model
model.load_state_dict(
    torch.load(
        "models/model.pth",
        weights_only=True
    )
)


# Evaluation mode
model.eval()


correct = 0
total = 0


# Disable gradient calculation
with torch.no_grad():

    for images, labels in test_loader:

        # Make predictions
        outputs = model(images)

        # Get predicted classes
        predictions = outputs.argmax(dim=1)

        # Count correct predictions
        correct += (predictions == labels).sum().item()

        # Count total samples
        total += labels.size(0)


# Calculate test accuracy
accuracy = 100 * correct / total


print(f"Test Accuracy: {accuracy:.2f}%")
