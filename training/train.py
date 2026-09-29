import torch
from explore_data import train_loader
from model import FashionCNN
from torch import nn

# Create model
model = FashionCNN()


# Loss function
loss_fn = nn.CrossEntropyLoss()


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training
for epoch in range(5):

    for images, labels in train_loader:

        # 1. Clear old gradients
        optimizer.zero_grad()

        # 2. Make predictions
        outputs = model(images)

        # 3. Calculate loss
        loss = loss_fn(outputs, labels)

        # 4. Calculate gradients
        loss.backward()

        # 5. Update model weights
        optimizer.step()

    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {loss.item():.4f}"
    )


# Save trained model
torch.save(
    model.state_dict(),
    "models/model.pth"
)

print("Model saved!")
