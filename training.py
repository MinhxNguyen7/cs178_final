import json
from pathlib import Path
from config import PREFERRED_DEVICE
from models import BaseModel

import torch
import torch.utils.data # I don't know why this isn't included in torch
import numpy as np

import shutil
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

def error_rate(model: BaseModel, loader: torch.utils.data.DataLoader, device: torch.device = PREFERRED_DEVICE) -> float:
    model.eval()
    model.to(device)

    total_correct = 0
    count = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            total_correct += torch.sum(predictions == labels).item()
            count += len(labels)

    return 1 - (total_correct / count)

def train(
    model: BaseModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device = PREFERRED_DEVICE,
    epochs: int = 1,
    log_interval: int = 100,
    results_file: str|Path|None = None,
    save_interval: int = 0,
    checkpoint_prefix: str = "checkpoint",
    verbose = True
) -> dict[str, np.ndarray]:
    """
    General implementation of the training loop. 
    
    Returns the training and validation losses as a 2-axis arrays of (epoch, log_point) in a dictionary.
    
    Saves the model every save_interval epochs if save_interval > 0. 
    Checkpoints are saved in {checkpoint_prefix}_{epoch}.pt.
    
    Saves the experiment results to results_file if results_file is not None.
    Validation error rates are calculated and saved to results["error_rates"] every epoch.
    
    Parameters:
        model: The model to train.
        train_loader: The training data loader.
        val_loader: The validation data loader.
        loss_fn: The loss function to use.
        device: The device to use for training.
        epochs: The number of epochs to train for.
        log_interval: The number of batches between each log.
        results_file: The path to save the experiment results to (json).
        save_interval: The number of epochs between each save. If 0, the model is not saved.
        checkpoint_prefix: The prefix to use for the checkpoint files.
        verbose: Whether to print the training and validation losses.
    
    Example: 
        losses = train(model, train_loader, val_loader, CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=0.0003), epochs = 10)
        
        training_loss = losses['train']
        validation_loss = losses['val']
        
        training_loss_per_epoch = np.mean(training_loss, axis=1)
        validation_loss_per_epoch = np.mean(validation_loss, axis=1)
    """
    if not model.optimizer:
        raise ValueError("Model has no optimizer. Did you forget to call model.apply_optimizer()?")
    
    print(f"Training on device {device}")
    terminal_width = shutil.get_terminal_size().columns
    
    model.train()
    model.to(device)
    
    results = {
        "setup": {
            "model": model.__class__.__name__,
            "optimizer": model.optimizer.__repr__(),
            "loss_fn": loss_fn.__repr__(),
        },
        "losses": {
            "train": np.zeros((epochs, len(train_loader))).tolist(),
            "val": np.zeros((epochs, len(train_loader))).tolist()
        },
        "error_rates": np.zeros(epochs).tolist()
    }
    
    if results_file:
        # Create the directory if it doesn't exist
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)

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

                results["losses"]["train"][epoch][batch] = training_loss
                results["losses"]["val"][epoch][batch] = validation_loss
                
                if not verbose: continue
                print(f"Batch {batch} of {len(train_loader)}: Training Loss = {training_loss}; Validation Loss = {validation_loss}")
                
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            checkpoint_path = Path(f"{checkpoint_prefix}_{epoch + 1}")
            model.save(f"{checkpoint_path}")
            print(f"Saved checkpoint to {checkpoint_path}.pt")
            
        if results_file:
            json.dump(results, open(results_file, "w"))
                
        end_time = time.time()
        
        print(f"Epoch {epoch + 1} took {round((end_time - start_time), 2)} seconds")
        print("-" * terminal_width) # Print a line to separate epochs
        

    return results["losses"]