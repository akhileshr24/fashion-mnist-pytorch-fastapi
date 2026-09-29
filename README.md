# Fashion MNIST Image Classifier API

A simple image classification project using PyTorch and FastAPI.

## Project Overview

This project trains a CNN model on the Fashion-MNIST dataset and exposes the trained model through a FastAPI REST API.

The API accepts a clothing image and returns:

- Predicted class
- Confidence score

## Classes

The model can classify:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Technologies

- Python
- PyTorch
- Torchvision
- FastAPI
- Uvicorn
- Pillow
- Scikit-learn
- Matplotlib

## Project Structure

```text
F:\hello\
│
├── app\
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── predict.py
│   └── schemas.py
│
├── training\
│   ├── __init__.py
│   ├── explore_data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── confusion_matrix.py
│   ├── classification_report.py
│   └── predict.py
│
├── data\
│   └── FashionMNIST\
│
├── models\
│   └── model.pth
│
├── requirements.txt
├── README.md
└── .gitignore
