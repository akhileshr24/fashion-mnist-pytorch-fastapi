import torch
from torch import nn


class FashionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # -------------------------
        # CNN Block 1
        # -------------------------
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # -------------------------
        # CNN Block 2
        # -------------------------
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Activation function
        self.relu = nn.ReLU()

        # Downsampling
        self.pool = nn.MaxPool2d(kernel_size=2)

        # -------------------------
        # Flatten
        # -------------------------
        self.flatten = nn.Flatten()

        # -------------------------
        # Fully Connected Layer
        # -------------------------
        self.fc = nn.Linear(
            in_features=32 * 7 * 7,
            out_features=10
        )

    def forward(self, x):

        # CNN Block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # CNN Block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Convert feature maps into one vector
        x = self.flatten(x)

        # Produce 10 class scores
        x = self.fc(x)

        return x


# Create the model
model = FashionCNN()

print(model)


# -------------------------
# Test the model
# -------------------------

# Fake batch of 64 Fashion-MNIST images
x = torch.randn(64, 1, 28, 28)

# Forward pass
output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Output:", output)

# Image
#   ↓
# Conv1       → Find basic features
#   ↓
# ReLU        → Add non-linearity / help learning
#   ↓
# MaxPool     → Reduce size, keep important information
#   ↓
# Conv2       → Find more complex features
#   ↓
# ReLU        → Help learn complex patterns
#   ↓
# MaxPool     → Reduce size again
#   ↓
# Flatten     → Convert all feature maps into ONE long vector
#   ↓
# Linear      → Use those features to classify
#   ↓
# 10 outputs  → 10 clothing classes
