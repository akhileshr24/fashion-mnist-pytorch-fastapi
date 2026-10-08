import torch
from sklearn.metrics import confusion_matrix

from data import test_loader
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


# Evaluation mode
model.eval()


# Store predictions and actual labels
predictions = []
actual = []


# Disable gradients
with torch.no_grad():

    for images, labels in test_loader:

        # Make predictions
        outputs = model(images)

        # Get predicted classes
        predicted_classes = outputs.argmax(dim=1)

        # Store results
        predictions.extend(predicted_classes.tolist())
        actual.extend(labels.tolist())


# Create confusion matrix
matrix = confusion_matrix(
    actual,
    predictions
)


print("Confusion Matrix:")
print(matrix)
