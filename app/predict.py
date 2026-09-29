import torch
from PIL import Image
from torchvision import transforms

from app.model import FashionCNN

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


model = FashionCNN()

model.load_state_dict(
    torch.load("models/model.pth", weights_only=True)
)

model.eval()


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


def predict_image(image: Image.Image):

    image = image.convert("L").resize((28, 28))
    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(output, dim=1)

        prediction = probabilities.argmax(dim=1).item()

        confidence = probabilities.max().item()

    return {
        "prediction": class_names[prediction],
        "confidence": confidence * 100
    }
