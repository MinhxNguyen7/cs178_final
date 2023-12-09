from torch.nn import Module, Linear, ReLU, Softmax, Flatten, Conv2d, MaxPool2d, Sequential, init
import torch

from collections import OrderedDict
from abc import ABC, abstractmethod

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

    def init_weights(self, module: Module) -> None:
        if not (isinstance(module, Conv2d) or isinstance(module, Linear)):
            return

        init.kaiming_normal_(module.weight)

        if module.bias is not None and callable(module.bias.data.fill_):
            module.bias.data.fill_(0)
            
