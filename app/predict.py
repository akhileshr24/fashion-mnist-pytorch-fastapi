import torch
from PIL import Image
from torchvision import transforms

from training.model import FashionCNN

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


# Create the model
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


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


def predict_image(image: Image.Image):

    # Convert image to grayscale
    image = image.convert("L")

    # Apply preprocessing
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    # Make prediction
    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = probabilities.argmax(
            dim=1
        ).item()

        confidence = probabilities.max().item()


    return {
        "prediction": class_names[prediction],
        "confidence": confidence * 100
    }
