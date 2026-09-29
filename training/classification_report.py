import torch
from sklearn.metrics import classification_report

from src.explore_data import test_dataset
from src.model import FashionCNN

# 1. Create the model
model = FashionCNN()


# 2. Load the trained model
model.load_state_dict(
    torch.load("model.pth")
)


# 3. Tell the model we are testing
model.eval()


# 4. Lists to store results
predictions = []
actual = []


# 5. Don't calculate gradients
with torch.no_grad():

    # Go through all test images
    for image, label in test_dataset:

        # Add batch dimension
        image = image.unsqueeze(0)

        # Ask the model for prediction
        output = model(image)

        # Get the class with the highest score
        prediction = output.argmax(dim=1).item()

        # Save model prediction
        predictions.append(prediction)

        # Save real answer
        actual.append(label)


# 6. Names of the 10 classes
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


# 7. Create classification report
report = classification_report(
    actual,
    predictions,
    target_names=class_names
)


# 8. Show the report
print(report)
