import torch
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

# Fake images
images = torch.randn(64, 1, 28, 28)

# Fake labels
labels = torch.randint(0, 10, (64,))


# -------------------------
# Training step
# -------------------------

# Clear old gradients
optimizer.zero_grad()

# Prediction
outputs = model(images)

# Calculate loss
loss = loss_fn(outputs, labels)

# Backpropagation
loss.backward()

# Update weights
optimizer.step()


print("Output shape:", outputs.shape)
print("Labels shape:", labels.shape)
print("Loss:", loss.item())
