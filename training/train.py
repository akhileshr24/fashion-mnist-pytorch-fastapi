import torch
from data import train_loader, val_loader
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


# Early stopping settings
num_epochs = 20
patience = 3

best_val_loss = float("inf")
patience_counter = 0


# Training
for epoch in range(num_epochs):

    # -------------------------
    # Training
    # -------------------------
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_fn(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    average_train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * train_correct / train_total


    # -------------------------
    # Validation
    # -------------------------
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            val_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    average_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total


    # -------------------------
    # Print results
    # -------------------------
    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {average_val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
    )


    # -------------------------
    # Save best model
    # -------------------------
    if average_val_loss < best_val_loss:

        best_val_loss = average_val_loss
        patience_counter = 0

        torch.save(
            model.state_dict(),
            "models/model.pth"
        )

        print("Best model saved!")

    else:

        patience_counter += 1

        print(
            f"No improvement. "
            f"Patience: {patience_counter}/{patience}"
        )

        if patience_counter >= patience:

            print("Early stopping triggered.")
            break


print("Training completed!")
