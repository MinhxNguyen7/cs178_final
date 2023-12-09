from config import PREFERRED_DEVICE
from models import BaseModel

import torch
import torch.utils.data # I don't know why this isn't included in torch
import numpy as np

import time

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
    model: BaseModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device = PREFERRED_DEVICE,
    epochs: int = 1,
    log_interval: int = 100
) -> dict[str, np.ndarray]:
    """
    General implementation of the training loop.
    
    Example: 
        losses = train(model, train_loader, val_loader, CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=0.0003, momentum=0.9), epochs = 10)
    """
    if not model.optimizer:
        raise ValueError("Model has no optimizer. Did you forget to call model.apply_optimizer()?")
    
    print(f"Training on device {device}")
    
    model.train()
    model.to(device)

    losses = {
        "train": np.zeros((epochs, len(train_loader))),
        "val": np.zeros((epochs, len(train_loader)))
    }

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        
        start_time = time.time()

        for batch, (images, labels) in enumerate(train_loader):
            # Send to device
            images = images.to(device)
            labels = labels.to(device)

            model.optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = loss_fn(outputs, labels)

            # Backward pass
            loss.backward()

            # Optimize
            model.optimizer.step()

            if batch % log_interval == 0:
                training_loss = torch.mean(loss).item()
                validation_loss = evaluate(model, val_loader, loss_fn, device)

                print(f"Batch {batch} of {len(train_loader)}: Training Loss = {training_loss}; Validation Loss = {validation_loss}")
                losses["train"][epoch, batch] = training_loss
                losses["val"][epoch, batch] = validation_loss
                
        end_time = time.time()
        print(f"Epoch {epoch + 1} took {end_time - start_time} seconds")

    return losses