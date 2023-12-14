from config import LEGEND_PATH, IMG_DIR
from transformations import cnn_preprocess

import torch.utils.data
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2

from typing import Callable
from pathlib import Path

class Dataset(torch.utils.data.Dataset):
    CLASSES = sorted(['anger', 'fear', 'sadness', 'neutral', 'happiness', 'surprise', 'contempt', 'disgust'])
    """
    Example:
        dataset = Dataset(legend, CLASSES, transformation)
        
        train_set, val_set, test_set = dataset.split([0.8, 0.1, 0.1])
        
        train_loader, val_loader, test_loader = map(dataloader_factory, [train_set, val_set, test_set])
    """
    def __init__(self, legend: pd.DataFrame, img_dir: str|Path, transform: Callable[[np.ndarray], np.ndarray] = lambda x: x):
        """
        Creates a dataset object for a legend of image paths and labels for use in PyTorch's Dataloader.
        
        Automatically drop unecessary columns and filter out images with unknown labels.

        Parameters
        ----------
        legend: pd.DataFrame
            A Pandas DataFrame with columns `image` and `emotion`.
        classes: list
            A list of classes to filter the legend by.
        transform: Callable[[np.ndarray], np.ndarray]
            A function to transform the raw image before converting it to a Tensor and sending it to the model.
        """
        self.img_dir = img_dir if isinstance(img_dir, Path) else Path(img_dir)
        
        self.legend = legend.copy()

        # Lowercase all labels
        self.legend['emotion'] = self.legend['emotion'].str.lower()

        # Filter out images with unknown labels
        self.legend = self.legend[self.legend['emotion'].isin(self.CLASSES)]

        # Reset indices
        self.legend = self.legend.reset_index(drop = True)

        # Drop unecessary columns
        self.legend = self.legend[['image', 'emotion']]

        self.transform = transform

    def __len__(self) -> int:
        return self.legend.shape[0]

    def split(self, sizes: list[float]) -> list["Dataset"]:
        """
        Probabilistically splits the dataset into subsets of given size ratios.

        Parameters
        ----------
        sizes: list[float]
            A list of size ratios to split the dataset into. Must sum to 1.
        """
        assert sum(sizes) == 1.0

        # Split Dataframes into partitions randomly by creating a new column with random values and filtering by them
        self.legend['partition'] = np.random.choice(
            range(len(sizes)),
            p = sizes,
            size = len(self)
        )

        dataframes = [self.legend[self.legend['partition'] == partition].copy() for partition in range(len(sizes))]

        # Drop the partition column
        for dataframe in dataframes:
            dataframe.drop(columns = ['partition'], inplace = True)

        # Reset the indices
        for dataframe in dataframes:
            dataframe.reset_index(drop = True, inplace = True)

        return [Dataset(dataframe, self.img_dir, self.transform) for dataframe in dataframes]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_name, label = self.legend.iloc[index].values

        img_path = str(Path(self.img_dir, img_name))

        # Read the image, convert it to a float tensor, and normalize it to [0, 1]
        img = torch.tensor(self.transform(cv2.imread(img_path))) / 255.0
        label = torch.tensor(self.CLASSES.index(label))

        return img, label

def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collates a list of tuples (image, label) into a batch of images and a batch of labels.
    """
    images, labels = zip(*batch)

    images = torch.stack(images)
    labels = torch.stack(labels)

    return images, labels

def dataloader_factory(dataset: Dataset, shuffle: bool = True, batch_size = 16) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for a given Dataset.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size,
        shuffle = shuffle,
        num_workers = 2,
        collate_fn=collate_fn,
        prefetch_factor=2
    )
    
def get_dataloaders(
    legend_path: str|Path = LEGEND_PATH,
    img_dir: str|Path = IMG_DIR,
    batch_size = 32,
    split = [0.8, 0.2]
):
    if not isinstance(legend_path, Path): legend_path = Path(legend_path)
    if not isinstance(img_dir, Path): img_dir = Path(img_dir)
    
    if not sum(split) == 1.0:
        raise ValueError("Split ratios must sum to 1.")
    
    # Data setup
    legend = pd.read_csv(legend_path)
    dataset = Dataset(legend, img_dir, transform=cnn_preprocess)
    split_datasets = dataset.split(split)

    return map(dataloader_factory, split_datasets)

if __name__ == '__main__':
    legend = pd.read_csv(LEGEND_PATH)
    dataset = Dataset(legend, IMG_DIR, cnn_preprocess)
    train_set, val_set, test_set = dataset.split([0.8, 0.1, 0.1])
    train_loader, val_loader, test_loader = map(dataloader_factory, [train_set, val_set, test_set])
    
    # Check that the transformation works properly
    sample_rows = train_set.legend.sample(4)
    fig, axes = plt.subplots(2, 4, figsize = (18, 8))

    for index, (img_name, label) in enumerate(
        sample_rows[["image", "emotion"]].values
    ):
        image = cv2.imread(str(Path(IMG_DIR, img_name)))
        axes[0, index].imshow(image)
        axes[0, index].set_title("before")

        transformed = cnn_preprocess(image)
        axes[1, index].imshow(transformed[0], cmap = "gray")
        axes[1, index].set_title("after")