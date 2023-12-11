import json
import shutil
import time
from config import PREFERRED_DEVICE

from torch.nn import Module, Linear, ReLU, Flatten, Conv2d, MaxPool2d, Sequential, init, Dropout
import torch.utils.data

import numpy as np
import torch

from collections import OrderedDict
from abc import ABC, abstractmethod
from pathlib import Path

from data_loading import Dataset


class BaseModel(Module, ABC):
    """
    Abstract class for models implementing the scaffolding.
    """
    
    def __init__(self, sequence: torch.nn.Sequential, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.sequence = sequence
        self.sequence.apply(self.init_weights)
        
        self._size = sum(p.numel() for p in self.sequence.parameters() if p.requires_grad)
        
        self.optimizer: None|torch.optim.Optimizer = None
        
    @abstractmethod
    def init_weights(self, module: Module) -> None:
        pass
    
    def forward(self, x):
        return self.sequence(x)
    
    def __call__(self, x):
        return self.forward(x)
    
    def predict(self, x):
        """
        Outputs the actual class name of the prediction.
        """
        return Dataset.CLASSES[int(torch.argmax(self(x)).item())]
    
    @property
    def size(self) -> int:
        return self._size
    
    def parameters(self):
        return self.sequence.parameters()
    
    def apply_optimizer(self, optimizer, **kwargs):
        """
        Example:
            model = LittleModel()
            model.apply_optimizer(torch.optim.SGD, lr = 0.0003, momentum = 0.9)
        """
        self.optimizer = optimizer(self.parameters(), **kwargs)
        self.optimizer
        
    def save(self, path: str|Path):
        torch.save(self.state_dict(), path)
        
    @staticmethod
    def from_checkpoint(path: str|Path):
        model = LittleModel()
        model.load_state_dict(torch.load(path))
        return model
    
    def evaluate(
        self,
        loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
        device: torch.device = PREFERRED_DEVICE
    ) -> float:
        self.eval()
        self.to(device)

        total_loss = 0.0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self(images)

                loss = loss_fn(outputs, labels)

                total_loss += loss.item()

        return total_loss / len(loader)

    def error_rate(self, loader: torch.utils.data.DataLoader, device: torch.device = PREFERRED_DEVICE) -> float:
        self.eval()
        self.to(device)

        total_correct = 0
        count = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self(images)

                predictions = torch.argmax(outputs, dim=1)

                total_correct += torch.sum(predictions == labels).item()
                count += len(labels)

        return 1 - (total_correct / count)
    
    def train_loop(
        self,
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
        if not self.optimizer:
            raise ValueError("Model has no optimizer. Did you forget to call model.apply_optimizer()?")
        
        print(f"Training on device {device}")
        terminal_width = shutil.get_terminal_size().columns
        
        self.to(device)
        
        results = {
            "setup": {
                "model": self.__class__.__name__,
                "optimizer": self.optimizer.__repr__(),
                "loss_fn": loss_fn.__repr__(),
            },
            "losses": {
                "train": np.zeros((epochs, len(train_loader)//log_interval)).tolist(),
                "val": np.zeros((epochs, len(train_loader)//log_interval)).tolist()
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
                self.train()
                
                # Send data to device
                images = images.to(device)
                labels = labels.to(device)

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self(images)

                # Calculate loss
                loss = loss_fn(outputs, labels)

                # Backward pass
                loss.backward()

                # Optimize
                self.optimizer.step()
                
                if batch % log_interval == 0:
                    training_loss = torch.mean(loss).item()
                    validation_loss = self.evaluate(val_loader, loss_fn, device)

                    results["losses"]["train"][epoch][batch//log_interval-1] = training_loss
                    results["losses"]["val"][epoch][batch//log_interval-1] = validation_loss
                    
                    if not verbose: continue
                    print(f"Batch {batch} of {len(train_loader)}: Training Loss = {training_loss}; Validation Loss = {validation_loss}")
                    
            if save_interval > 0 and (epoch + 1) % save_interval == 0:
                checkpoint_path = Path(f"{checkpoint_prefix}_{epoch + 1}.pt")
                self.save(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                
            if results_file:
                results["error_rates"][epoch] = self.error_rate(val_loader, device)
                json.dump(results, open(results_file, "w"))
                    
            end_time = time.time()
            
            print(f"Epoch {epoch + 1} took {round((end_time - start_time), 2)} seconds")
            print("-" * terminal_width) # Print a line to separate epochs
            

        return results["losses"]
    
class LittleModel(BaseModel):
    """
    Takes a greyscale image and predicts one of the eight classes.
    """
    
    def __init__(self):
        # Input shape = (1, 350, 350)

        layers: OrderedDict[str, Module] = OrderedDict([
            ("conv1", Conv2d(1, 64, 5, 2)), # (64, 263, 263)
            ("pool1", MaxPool2d(kernel_size = 5, stride=2)), # (64, 130, 130)

            ("conv2", Conv2d(64, 256, 3)), # (256, 128, 128)
            ("pool2", MaxPool2d(kernel_size = 3, stride = 2, padding = 1)), # (256, 64, 64)

            ("conv3", Conv2d(256, 512, 3, 1, padding = 1)), # (512, 64, 64)
            ("pool3", MaxPool2d(3, 2, 1)), # (512, 32, 32)

            ("conv4", Conv2d(512, 1024, 3, 1, padding = 1)), # (512, 32, 32)
            ("pool4", MaxPool2d(3, 2, 1)), # (512, 16, 16)

            ("conv5", Conv2d(1024, 1024, 3, 1, padding = 1)), # (1024, 14, 14)
            ("pool5", MaxPool2d(3, 2, 1)), # (1024, 6, 6)

            # Fully-connected
            ("flat", Flatten()),

            ("fc1", Linear(1024 * 6 * 6,  128)),
            ("relu1", ReLU()),

            ("fc2", Linear(128, 32)),
            ("relu2", ReLU()),

            ("fc3", Linear(32, 8)),
        ])
        
        super().__init__(Sequential(layers))

    @staticmethod
    def init_weights(module: Module) -> None:
        if not (isinstance(module, Conv2d) or isinstance(module, Linear)):
            return

        init.kaiming_normal_(module.weight)

        if module.bias is not None and callable(module.bias.data.fill_):
            module.bias.data.fill_(0)
    
    @staticmethod
    def create_default():
        """
        Create the default model to skip the setup.
        """
        loss = torch.nn.CrossEntropyLoss()
        model = LittleModel()
        model.apply_optimizer(torch.optim.Adam, lr=0.0001)
        
        return model, loss

class MoreFCDropout(BaseModel):
    """
    Same as little model but with more fully-connected layers. 
    This can potentially help with convergence because not so much information needs to pass through each neuron.
    Then, we can apply dropout to prevent overfitting.
    """
    
    def __init__(self, dropout_rate = 0.0):
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate must be between 0 and 1")
        
        # Input shape = (1, 350, 350)

        layers: OrderedDict[str, Module] = OrderedDict([
            ("conv1", Conv2d(1, 64, 5, 2)), # (64, 263, 263)
            ("pool1", MaxPool2d(kernel_size = 5, stride=2)), # (64, 130, 130)

            ("conv2", Conv2d(64, 256, 3)), # (256, 128, 128)
            ("pool2", MaxPool2d(kernel_size = 3, stride = 2, padding = 1)), # (256, 64, 64)

            ("conv3", Conv2d(256, 512, 3, 1, padding = 1)), # (512, 64, 64)
            ("pool3", MaxPool2d(3, 2, 1)), # (512, 32, 32)

            ("conv4", Conv2d(512, 1024, 3, 1, padding = 1)), # (512, 32, 32)
            ("pool4", MaxPool2d(3, 2, 1)), # (512, 16, 16)

            ("conv5", Conv2d(1024, 1024, 3, 1, padding = 1)), # (1024, 14, 14)
            ("pool5", MaxPool2d(3, 2, 1)), # (1024, 6, 6)

            # Fully-connected
            ("flat", Flatten()),

            ("drop1", Dropout(dropout_rate)),
            ("fc1", Linear(1024 * 6 * 6,  2048)),
            ("relu1", ReLU()),

            ("drop2", Dropout(dropout_rate)),
            ("fc2", Linear(2048, 512)),
            ("relu2", ReLU()),
            
            ("drop3", Dropout(dropout_rate)),
            ("fc3", Linear(512, 128)),
            ("relu3", ReLU()),

            ("drop4", Dropout(dropout_rate)),
            ("fc4", Linear(128, 64)),
            ("relu4", ReLU()),

            # 64 nodes is already quite restrictive, so we probably don't need dropout here
            ("fc6", Linear(64, 8)),
        ])
        
        super().__init__(Sequential(layers))

    @staticmethod
    def init_weights(module: Module) -> None:
        if not (isinstance(module, Conv2d) or isinstance(module, Linear)):
            return

        init.kaiming_normal_(module.weight)

        if module.bias is not None and callable(module.bias.data.fill_):
            module.bias.data.fill_(0)
    
    @staticmethod
    def create_default():
        """
        Create the default model to skip the setup.
        """
        loss = torch.nn.CrossEntropyLoss()
        model = LittleModel()
        model.apply_optimizer(torch.optim.Adam, lr=0.0001, weight_decay=0.0005)
        
        return model, loss