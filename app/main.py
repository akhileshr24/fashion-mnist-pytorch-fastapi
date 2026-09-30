import logging
from io import BytesIO

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app.schemas import PredictionResponse
from training.model import FashionCNN

app = FastAPI(
    title="Fashion MNIST Classifier API",
    description="Image classification using PyTorch CNN",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


# Load trained model
model = FashionCNN()

model.load_state_dict(
    torch.load(
        "models/model.pth",
        weights_only=True
    )
)

model.eval()
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(file: UploadFile = File(...)):

    # Log incoming request
    logger.info(
        f"Prediction request: {file.filename}"
    )

    # Validate file type
    allowed_types = [
        "image/jpeg",
        "image/png"
    ]

    if file.content_type not in allowed_types:
        logger.warning(
            f"Invalid file type: {file.content_type}"
        )

        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are allowed."
        )

    # Validate file size
    MAX_FILE_SIZE = 5 * 1024 * 1024

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        logger.warning(
            f"File too large: {len(contents)} bytes"
        )

        raise HTTPException(
            status_code=400,
            detail="File size must be less than 5 MB."
        )

    # Convert bytes → PIL Image
    try:
        image = Image.open(
            BytesIO(contents)
        )

    except UnidentifiedImageError:

        logger.warning(
            "Uploaded file is not a valid image"
        )

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    # Preprocess image
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    # Make prediction
    try:
        with torch.no_grad():

            output = model(image)

            # Convert logits → probabilities
            probabilities = torch.softmax(
                output,
                dim=1
            )

            # Get highest probability class
            prediction = probabilities.argmax(
                dim=1
            ).item()

            # Convert class number → class name
            predicted_class = class_names[prediction]

            # Get confidence
            confidence = probabilities.max().item()

    except Exception:

        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try another image."
        )

    # Log prediction
    logger.info(
        f"Prediction: {predicted_class} | "
        f"Confidence: {confidence * 100:.2f}%"
    )

    return {
        "prediction": predicted_class,
        "confidence": confidence * 100
    }
