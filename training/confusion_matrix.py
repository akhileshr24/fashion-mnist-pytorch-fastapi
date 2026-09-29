import torch
from sklearn.metrics import confusion_matrix

from src.explore_data import test_dataset
from src.model import FashionCNN

model = FashionCNN()

model.load_state_dict(torch.load("model.pth"))

model.eval()

predictions=[]
actual = []

with torch.no_grad():
    for image, label in test_dataset:

        image =  image.unsqueeze(0)

        output = model(image)

        prediction = output.argmax(dim=1).item()

        # if prediction == label:
        #     print("Correct!!")
        # else:
        #     print("Wrong!")

        predictions.append(prediction)
        actual.append(label)


matrix = confusion_matrix(actual, predictions)

print(matrix)
