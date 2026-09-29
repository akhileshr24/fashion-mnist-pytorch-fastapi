import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.predict import predict_image
from app.schemas import PredictionResponse

app = FastAPI(
    title="Fashion MNIST Classifier API",
    description="Fashion image classification using PyTorch",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Fashion MNIST API is running"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):

    logger.info(f"Prediction request received: {file.filename}")

    # 1. Check file type
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]:
        logger.warning(f"Invalid file type: {file.content_type}")

        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are allowed."
        )

    # 2. Check file size
    contents = await file.read()

    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image size must be less than 5 MB."
        )

    # 3. Open and validate image
    try:
        image = Image.open(file.file)

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    # 4. Run prediction
    try:
        result = predict_image(image)

        logger.info(
            f"Prediction: {result['prediction']} | "
            f"Confidence: {result['confidence']:.2f}%"
        )

        return result

    except Exception:

        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try another image."
        )
