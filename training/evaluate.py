import torch
from explore_data import test_dataset
from model import FashionCNN

# Create model
model = FashionCNN()


# Load trained weights
model.load_state_dict(
    torch.load(
        "models/model.pth",
        weights_only=True
    )
)


# Put model in evaluation mode
model.eval()


correct = 0
total = 0


# Disable gradient calculation
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

        # Check prediction
        if prediction == label:
            correct += 1

        total += 1


# Calculate accuracy
accuracy = 100 * correct / total

print(
    f"Test Accuracy: {accuracy:.2f}%"
)
