import torch
import torch.nn


class FashionCNN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.relu = torch.nn.ReLU()

        self.pool = torch.nn.MaxPool2d(
            kernel_size=2
        )

        self.flatten = torch.nn.Flatten()

        self.fc = torch.nn.Linear(
            in_features=32 * 7 * 7,
            out_features=10
        )


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
