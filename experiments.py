"""
File to keep track of experiments and hyperparameter tuning.
"""

from pathlib import Path
import shutil
from typing import Any, Iterable
import pandas as pd
import torch

from config import CHECKPOINTS_DIR, IMG_DIR, LEGEND_PATH, RESULTS_DIR
from data_loading import Dataset, dataloader_factory, get_dataloaders
from models import LittleModel, MoreFCDropout, TorchModel
from transformations import cnn_preprocess
from visualizations import visualize_dropout_variations, visualize_l2_variations, visualize_losses, visualize_lr_scheduler_variations

def scheduler_repr(scheduler: None|type[torch.optim.lr_scheduler.LRScheduler], **kwargs):
    if scheduler is None:
        return "None"
    return f"{scheduler.__name__}({', '.join([f'{k}={v}' for k, v in kwargs.items()])})"

def lr_scheduler_variations(
    experiment_name: str = "lr_schedulers", 
    schedulers: list[None|type[torch.optim.lr_scheduler.LRScheduler]] = [
        None, 
        torch.optim.lr_scheduler.ExponentialLR
    ],
    scheduler_kwargs_list: list[dict[str, Any]] = [
        {},
        {"gamma": 0.9, "last_epoch": 50}
    ],
    initial_lrs: list[float] = [0.001, 0.0005, 0.0001],
    models: Iterable[TorchModel] = (MoreFCDropout(0.3), LittleModel()),
    epochs = 50, 
    verbose = False,
    visualize = False,
):
    """
    Test learning rate schedulers with MoreFCDropout(0.3) and LittleModel.
    
    Experiments with different learning rate schedulers, step sizes, and gamma values.
    """
    results = pd.DataFrame(columns=["model", "lr_scheduler", "initial_lr", "train_loss", "test_loss", "test_error"])
    
    train_loader, test_loader = get_dataloaders()
    
    for model in models:
        for scheduler_index, scheduler in enumerate(schedulers):
            scheduler_kwargs = scheduler_kwargs_list[scheduler_index]
            
            for lr in initial_lrs:
                model, perf = lr_scheduler_experiment(
                    model, scheduler, scheduler_kwargs, lr, train_loader, test_loader, epochs, experiment_name, verbose
                )
                
                # Save results
                results = pd.concat(
                    [
                        results, 
                        pd.DataFrame(
                            [[model.__repr__(), scheduler_repr(scheduler, **scheduler_kwargs), lr, perf["train"], perf["val"], perf["error_rates"]]], 
                            columns=results.columns
                        )
                    ], 
                    ignore_index=True
                )
                    
    # Save results to file
    results.to_csv(Path(RESULTS_DIR, experiment_name + ".csv"), index=False)
    
    visualize_lr_scheduler_variations(results, visualize)
    
    return results

def lr_scheduler_experiment(
    model: TorchModel = MoreFCDropout(0.3),
    scheduler: None|type[torch.optim.lr_scheduler.LRScheduler] = None,
    scheduler_kwargs: dict = {},
    initial_lr: float = 0.00015,
    train_loader = None,
    test_loader = None,
    epochs = 50,
    experiment_name: str = "lr_scheduler_experiment",
    verbose = True
):
    """
    Experiment with a learning rate scheduler.
    """
    experiment_name += f"-{model}"
    if scheduler is not None:
        scheduler_name = scheduler.__name__
        experiment_name += f"-{scheduler_name}"
        
        experiment_name += f"-{scheduler_repr(scheduler, **scheduler_kwargs)}"
    
    else:
        experiment_name += f"-None"
    
    if train_loader is None or test_loader is None:
        train_loader, test_loader = get_dataloaders()
    
    # Model setup
    model.apply_optimizer(torch.optim.Adam, lr=initial_lr)
    if scheduler is not None:
        model.apply_scheduler(scheduler, **scheduler_kwargs)
        
    loss = torch.nn.CrossEntropyLoss()
    
    # Training
    perf = model.train_loop(
        train_loader, test_loader, loss, epochs=epochs, 
        results_file=Path(RESULTS_DIR, experiment_name + ".json"), save_interval=min(10, epochs), 
        checkpoint_prefix=experiment_name, verbose=verbose
    )
    
    return model, perf

def more_fc_dropout_variations(
    dropouts: list[float] = [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    experiment_name: str = 'more_fc_dropout',
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
            epochs=epochs, results_file=Path(RESULTS_DIR, experiment_name + f"={dropout}.json"),
            save_interval=0, verbose=verbose
        )
        
        model_save_path = Path(CHECKPOINTS_DIR, experiment_name + f"={dropout}.pt")
        model.save(model_save_path)
        print(f"Saved model to {model_save_path}")
        
    visualize_dropout_variations(experiment_name, dropouts)
    

def little_model_l2_variations(
    # There's a typo here, it should be 5e-3 instead of 5e-2
    decays: list[float] = [0, 1e-4, 2e-4, 5e-4, 1e-3, 5e-2, 1e-2, 1e-1], 
    experiment_name: str = "little_model_decay",
    epochs = 50
):
    for decay in decays:
        print(f"Training model with decay={decay}".center(shutil.get_terminal_size().columns, "="))
        # Will automatically save results and losses
        little_model_with_l2(decay, experiment_name + f"={decay}", epochs, visualize = False)
        
    visualize_l2_variations(experiment_name, decays)
    

def little_model_with_l2(decay = 0.0001, experiment_name: str = "little_model_with_l2", epochs = 50, visualize = True):
    train_loader, test_loader = get_dataloaders()

    # Model setup
    model = LittleModel()
    loss = torch.nn.CrossEntropyLoss()
    model.apply_optimizer(torch.optim.Adam, lr=0.00015, weight_decay=decay)
    
    # Training
    losses = model.train_loop(
        train_loader, test_loader, loss, 
        epochs=epochs, results_file=Path(RESULTS_DIR, experiment_name + ".json"), 
        save_interval=10, checkpoint_prefix=experiment_name
    )
    
    if visualize:
        # Visualize and save visualizations to file
        visualize_losses(losses["train"], losses["val"], Path(RESULTS_DIR, experiment_name + ".png"))
        
    return model, losses


def little_model(experiment_name: str = "little_model", epochs = 50, visualize = True):
    BATCH_SIZE = 32
    experiment_name = "little_model"

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
        epochs=epochs, results_file=Path(RESULTS_DIR, experiment_name + ".json"), 
        save_interval=10, checkpoint_prefix=experiment_name
    )
    
    if visualize:
        # Visualize and save visualizations to file
        visualize_losses(losses["train"], losses["val"], Path(RESULTS_DIR, experiment_name + ".png"))
        
    return model, losses

if __name__ == "__main__":
    # more_fc_dropout_variations()
    # visualize_dropout_variations("more_fc_dropout", [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    
    lr_scheduler_variations(
        schedulers=[None, torch.optim.lr_scheduler.ExponentialLR], 
        initial_lrs=[0.00015],
        scheduler_kwargs_list=[{}, {"gamma": 0.9}],
        epochs=30,
    )
