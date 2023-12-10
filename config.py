import torch
import numpy as np

from pathlib import Path
import random

GPU = torch.device("cuda:0") if torch.cuda.is_available() else None
CPU = torch.device("cpu")
PREFERRED_DEVICE = GPU if GPU else CPU

LEGEND_PATH = Path("facial_expressions/data/legend.csv")
IMG_DIR = Path("facial_expressions/images")

CHECKPOINTS_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")

SEED = 1234

np.random.seed(SEED)
random.seed(SEED)