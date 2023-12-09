from config import PREFERRED_DEVICE

import torch
import torch.utils.data # I don't know why this isn't included in torch
import numpy as np



def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
    device: torch.device = PREFERRED_DEVICE
) -> float:
    model.eval()
    model.to(device)

    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

    return total_loss / len(loader)

def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = PREFERRED_DEVICE,
    epochs: int = 1,
    log_interval: int = 100
) -> dict[str, np.ndarray]:
    """
    General implementation of the training loop.
    
    Example: 
        losses = train(model, train_loader, val_loader, CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=0.0003, momentum=0.9), epochs = 10)
    """
    model.train()
    model.to(device)

    losses = {
        "train": np.zeros((epochs, len(train_loader))),
        "val": np.zeros((epochs, len(train_loader)))
    }

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        for batch, (images, labels) in enumerate(train_loader):
            # Send to device
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = loss_fn(outputs, labels)

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            if batch % log_interval == 0:
                training_loss = torch.mean(loss).item()
                validation_loss = evaluate(model, val_loader, loss_fn, device)

                print(f"Batch {batch} of {len(train_loader)}: Training Loss = {training_loss}; Validation Loss = {validation_loss}")
                losses["train"][epoch, batch] = training_loss
                losses["val"][epoch, batch] = validation_loss

    return losses