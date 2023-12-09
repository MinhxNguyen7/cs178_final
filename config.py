import torch
from pathlib import Path

GPU = torch.device("cuda:0") if torch.cuda.is_available() else None
CPU = torch.device("cpu")
PREFERRED_DEVICE = GPU if GPU else CPU

print(f"Using {PREFERRED_DEVICE}")

LEGEND_PATH = Path("facial_expressions/data/legend.csv")
IMG_DIR = Path("facial_expressions/data/images")