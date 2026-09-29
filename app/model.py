from torch import nn


class FashionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(32 * 7 * 7, 10)


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x
