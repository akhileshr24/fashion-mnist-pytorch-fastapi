import torch

from src.explore_data import test_dataset
from src.model import FashionCNN

# Create model
model = FashionCNN()

# Load trained model -> Puts those weights into our CNN.
model.load_state_dict(torch.load("model.pth"))

# Evaluation mode
model.eval()


# Test images
correct = 0
total = 0

# Disable gradient calculation -> testing no change in model
with torch.no_grad():

    for image, label in test_dataset:

        # Add batch dimension
        image = image.unsqueeze(0)

        # Prediction
        output = model(image)

        # Get predicted class
        prediction = output.argmax(dim=1).item()

        # Check prediction
        if prediction == label:
            correct += 1

        total += 1


accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")
