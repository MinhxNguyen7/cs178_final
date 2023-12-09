"""
Utility functions for testing.
"""

from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd

from config import IMG_DIR, LEGEND_PATH
from data_loading import Dataset, dataloader_factory

import torch.utils.data

from transformations import cnn_preprocess

def single_dataloaders(
    legend_path: str|Path = LEGEND_PATH, 
    img_dir: str|Path = IMG_DIR, 
    transform: Callable[[np.ndarray], np.ndarray] = cnn_preprocess
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create two dataloaders, each with one datapoint.
    """
    
    # Only the first datapoint
    train_legend = pd.read_csv(legend_path)
    train_legend = train_legend.drop(train_legend.index[1:])

    # Only the second datapoint
    val_legend = pd.read_csv(legend_path)
    val_legend = val_legend.drop(val_legend.index[0] + val_legend.index[2:])

    # Create dataset and dataloader
    train_dataset = Dataset(train_legend, img_dir, transform)
    val_dataset = Dataset(val_legend, img_dir, transform)
    train_dataloader = dataloader_factory(train_dataset, batch_size=1, shuffle=False)
    val_dataloader = dataloader_factory(val_dataset, batch_size=1, shuffle=False)
    
    return train_dataloader, val_dataloader
    
    