"""
File to keep track of experiments and hyperparameter tuning.
"""

from pathlib import Path
import shutil
import pandas as pd
import torch

from config import CHECKPOINTS_DIR, IMG_DIR, LEGEND_PATH, RESULTS_DIR
from data_loading import Dataset, dataloader_factory, get_dataloaders
from models import LittleModel, MoreFCDropout
from transformations import cnn_preprocess
from visualizations import visualize_dropout_variations, visualize_l2_variations, visualize_losses


def more_fc_dropout_variations(
    dropouts: list[float] = [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    model_name: str = 'more_fc_dropout',
    epochs = 75, # More epochs for a deeper, (hopefully) more stable model
    verbose = False
):
    train_loader, test_loader = get_dataloaders()
    
    for dropout in dropouts:
        print(f"Training model with dropout={dropout}".center(shutil.get_terminal_size().columns, "="))
        
        model = MoreFCDropout(dropout)
        model.apply_optimizer(torch.optim.Adam, lr=0.00015, weight_decay=0.0005)
        loss = torch.nn.CrossEntropyLoss()
        
        # Will automatically save results and losses
        model.train_loop(
            train_loader, test_loader, loss,
            epochs=epochs, results_file=Path(RESULTS_DIR, model_name + f"={dropout}.json"),
            save_interval=0, verbose=verbose
        )
        
        model_save_path = Path(CHECKPOINTS_DIR, model_name + f"={dropout}.pt")
        model.save(model_save_path)
        print(f"Saved model to {model_save_path}")
        
    visualize_dropout_variations(model_name, dropouts)
    

def little_model_l2_variations(
    # There's a typo here, it should be 5e-3 instead of 5e-2
    decays: list[float] = [0, 1e-4, 2e-4, 5e-4, 1e-3, 5e-2, 1e-2, 1e-1], 
    model_name: str = "little_model_decay",
    epochs = 50
):
    for decay in decays:
        print(f"Training model with decay={decay}".center(shutil.get_terminal_size().columns, "="))
        # Will automatically save results and losses
        little_model_with_l2(decay, model_name + f"={decay}", epochs, visualize = False)
        
    visualize_l2_variations(model_name, decays)
    

def little_model_with_l2(decay = 0.0001, model_name: str = "little_model_with_l2", epochs = 50, visualize = True):
    train_loader, test_loader = get_dataloaders()

    # Model setup
    model = LittleModel()
    loss = torch.nn.CrossEntropyLoss()
    model.apply_optimizer(torch.optim.Adam, lr=0.00015, weight_decay=decay)
    
    # Training
    losses = model.train_loop(
        train_loader, test_loader, loss, 
        epochs=epochs, results_file=Path(RESULTS_DIR, model_name + ".json"), 
        save_interval=10, checkpoint_prefix=model_name
    )
    
    if visualize:
        # Visualize and save visualizations to file
        visualize_losses(losses["train"], losses["val"], Path(RESULTS_DIR, model_name + ".png"))
        
    return model, losses


def little_model(model_name: str = "little_model", epochs = 50, visualize = True):
    BATCH_SIZE = 32
    model_name = "little_model"

    # Data setup
    legend = pd.read_csv(LEGEND_PATH)
    dataset = Dataset(legend, IMG_DIR, transform=cnn_preprocess)
    train_data, test_data = dataset.split([0.8, 0.2])

    train_loader = dataloader_factory(train_data, batch_size=BATCH_SIZE)
    test_loader = dataloader_factory(test_data, batch_size=BATCH_SIZE)

    # Model setup
    model = LittleModel()
    loss = torch.nn.CrossEntropyLoss()
    model.apply_optimizer(torch.optim.Adam, lr=0.00015)
    
    # Training
    losses = model.train_loop(
        train_loader, test_loader, loss, 
        epochs=epochs, results_file=Path(RESULTS_DIR, model_name + ".json"), 
        save_interval=10, checkpoint_prefix=model_name
    )
    
    if visualize:
        # Visualize and save visualizations to file
        visualize_losses(losses["train"], losses["val"], Path(RESULTS_DIR, model_name + ".png"))
        
    return model, losses

if __name__ == "__main__":
    more_fc_dropout_variations()
