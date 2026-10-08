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
import torch


class FashionCNN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # First convolution
        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # Second convolution
        self.conv2 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Activation function
        self.relu = torch.nn.ReLU()

        # Reduce image size by 2x
        self.pool = torch.nn.MaxPool2d(
            kernel_size=2
        )

        # Convert feature maps into a vector
        self.flatten = torch.nn.Flatten()

        # Final classification layer
        self.fc = torch.nn.Linear(
            in_features=32 * 7 * 7,
            out_features=10
        )


    def forward(self, x):

        # 1 × 28 × 28
        x = self.conv1(x)

        # 16 × 28 × 28
        x = self.relu(x)

        # 16 × 14 × 14
        x = self.pool(x)

        # 32 × 14 × 14
        x = self.conv2(x)

        # 32 × 14 × 14
        x = self.relu(x)

        # 32 × 7 × 7
        x = self.pool(x)

        # 32 × 7 × 7 → 1568
        x = self.flatten(x)

        # 1568 → 10 logits
        x = self.fc(x)

        return x
