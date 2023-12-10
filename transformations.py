"""
Image transformations for formatting and data augmentation.
"""

import numpy as np
import cv2

def cnn_preprocess(img: np.ndarray) -> np.ndarray:
    """
    Transform a raw image of shape (h, w, 3) to a square, greyscale image of shape (1, 350, 350).
    Convert it to grayscale, then pad the smaller dimension, then resize it with `cv2.resize`.
    """
    return greyscale(square(augment(img)))

def greyscale(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to greyscale. Shape (h, w, 3) -> (1, h, w).
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape((1, 350, 350))

def square(img: np.ndarray, size: int = 350, interpolation = cv2.INTER_CUBIC) -> np.ndarray:
    """
    Transform a raw image of shape (h, w, c) to a square image of (size, size, c).
    
    If the image is of shape (h, w), then the output will be of shape (1, size, size) to allow for batching with PyTorch.
    
    Uses cv2.copyMakeBorder to pad the image and cv2.resize for interpolation it.
    """
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else None
    
    # Pad the image to make it square
    if height > width:
        # Pad left and right
        padding = (height - width) // 2
        img = cv2.copyMakeBorder(img, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value = (0, 0, 0))
    elif width > height:
        # Pad top and bottom
        padding = (width - height) // 2
        img = cv2.copyMakeBorder(img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value = (0, 0, 0))

    # Resize to 350x350
    img = cv2.resize(img, (size, size), interpolation = interpolation)
    
    if channels is None:
        # Reshape the image so that it's (1, size, size)
        img = img.reshape((1, size, size))
    
    return img

# TODO: Implement data augmentation
def augment(img: np.ndarray) -> np.ndarray:
    return img
    