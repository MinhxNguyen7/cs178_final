import json
import math
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import pandas as pd
import torch
from config import CHECKPOINTS_DIR, RESULTS_DIR
from models import BaseModel, LittleModel


def visualize_losses(train: np.ndarray, test: np.ndarray, save_path: str|Path, show = True):
    plt.plot(train, label="train")
    plt.plot(test, label="test")
    plt.legend()
    plt.savefig(save_path)
    
    if not show: plt.close()

def visualize_l2_variations(model_name: str, decays: list[float], show = True):
    # Retrieve training results
    results = pd.DataFrame(columns=["decay", "train_loss", "test_loss", "test_error"])
    for decay in decays:
        with open(Path(RESULTS_DIR, model_name + f"={decay}.json"), 'r') as f:
            result = json.load(f)
        
        training_losses = np.array(result["losses"]["train"]).mean(axis=1) # Average loss per epoch
        test_losses = np.array(result["losses"]["val"]).mean(axis=1) # Average loss per epoch
        test_errors = np.array(result["error_rates"]) # Already per epoch
        
        results = pd.concat(
            [results, pd.DataFrame([[decay, training_losses, test_losses, test_errors]], columns=results.columns)], 
            ignore_index=True
        )

    # Calculate number of rows and columns such that the number of subplots is as close to a square as possible
    num_subplots = len(decays)
    num_rows = math.floor(math.sqrt(num_subplots))
    num_cols = math.ceil(num_subplots / num_rows) 

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))
    fig.suptitle("Training and validation losses per epoch for different decays")

    # Figure where each subplot is the training and validation loss of each decay
    for index, decay in enumerate(decays):
        row, col = divmod(index, num_cols)
        
        # Plot training and validation losses
        axs[row, col].plot(results.loc[index, "train_loss"], label="train")
        axs[row, col].plot(results.loc[index, "test_loss"], label="test")
        axs[row, col].set_title(f"decay={decay}")
        axs[row, col].legend()
        axs[row, col].set_ylim(0, 0.1)
        
    plt.savefig(Path(RESULTS_DIR, model_name + "_losses.png"))
    if not show: plt.close()

    # Plot test errors
    plt.figure(figsize=(20, 10))
    plt.suptitle("Test error per epoch for different decays")
    for index, decay in enumerate(decays):
        plt.plot(results.loc[index, "test_error"], label=f"decay={decay}")
        
    plt.savefig(Path(RESULTS_DIR, model_name + "_errors.png"))
    if not show: plt.close()

def visualize_dropout_variations(model_name: str, dropouts: list[float] = [0, 0.1, 0.2, 0.3, 0.4, 0.5], show = True):
    # Retrieve training results
    results = pd.DataFrame(columns=["dropout", "train_loss", "test_loss", "test_error"])
    for dropout in dropouts:
        with open(Path(RESULTS_DIR, model_name + f"={dropout}.json"), 'r') as f:
            result = json.load(f)
        
        training_losses = np.array(result["losses"]["train"]).mean(axis=1) # Average loss per epoch
        test_losses = np.array(result["losses"]["val"]).mean(axis=1) # Average loss per epoch
        test_errors = np.array(result["error_rates"]) # Already per epoch
        
        results = pd.concat(
            [results, pd.DataFrame([[dropout, training_losses, test_losses, test_errors]], columns=results.columns)], 
            ignore_index=True
        )

    # Calculate number of rows and columns such that the number of subplots is as close to a square as possible
    num_subplots = len(dropouts)
    num_rows = math.floor(math.sqrt(num_subplots))
    num_cols = math.ceil(num_subplots / num_rows) 

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))
    fig.suptitle("Training and validation losses per epoch for different dropouts")

    # Figure where each subplot is the training and validation loss of each dropout
    for index, dropout in enumerate(dropouts):
        row, col = divmod(index, num_cols)
        
        # Plot training and validation losses
        axs[row, col].plot(results.loc[index, "train_loss"], label="train")
        axs[row, col].plot(results.loc[index, "test_loss"], label="test")
        axs[row, col].set_title(f"dropout={dropout}")
        axs[row, col].legend()
        axs[row, col].set_ylim(0, 0.1)
    
    plt.legend()
    plt.savefig(Path(RESULTS_DIR, model_name + "_losses.png"))
    if not show: plt.close()

    # Plot test errors
    plt.figure(figsize=(20, 10))
    plt.suptitle("Test error per epoch for different dropouts")
    for index, dropout in enumerate(dropouts):
        plt.plot(results.loc[index, "test_error"], label=f"dropout={dropout}")
    
    plt.legend()
    plt.savefig(Path(RESULTS_DIR, model_name + "dropout_errors.png"))
    if not show: plt.close()
