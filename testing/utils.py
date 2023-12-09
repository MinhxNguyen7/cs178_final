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

def rowwise_equality(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """
    Returns True if all row values in a Dataframe are equal.
    """
    for a_series, b_series in zip(a.iterrows(), b.iterrows()):
        if not all(x == y for x, y in zip(a_series[1].values, b_series[1].values)):
            return False
        
    return True

def single_dataloaders(
    legend_path: str|Path = LEGEND_PATH, 
    img_dir: str|Path = IMG_DIR, 
    label_name = 'emotion',
    transform: Callable[[np.ndarray], np.ndarray] = cnn_preprocess
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create two dataloaders, each with one datapoint with a different label.
    """
    
    # Only the first datapoint
    train_legend = pd.read_csv(legend_path)
    train_legend.drop(index=train_legend.index[1:], inplace=True)

    # Find the first data point with a different label
    val_legend = pd.read_csv(legend_path)
    for i, label in enumerate(val_legend[label_name]):
        if label != train_legend[label_name].iloc[0]:
            val_legend.drop(index=val_legend.index[i+1:], inplace=True) # Drop all datapoints after the first different label
            val_legend.drop(index=val_legend.index[:i], inplace=True) # Drop all datapoints before the first different label
            break
    else:
        # If no different label was found
        raise ValueError('No different label was found in the legend.')
    
    assert not rowwise_equality(train_legend, val_legend)
    
    # Create dataset and dataloader
    train_dataset = Dataset(train_legend, img_dir, transform)
    val_dataset = Dataset(val_legend, img_dir, transform)
    train_dataloader = dataloader_factory(train_dataset, batch_size=1, shuffle=False)
    val_dataloader = dataloader_factory(val_dataset, batch_size=1, shuffle=False)
    
    return train_dataloader, val_dataloader
    
    