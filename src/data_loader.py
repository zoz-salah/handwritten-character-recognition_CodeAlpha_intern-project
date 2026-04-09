"""
Dataset loading and preprocessing for MNIST and EMNIST.
TensorFlow/Keras provides convenient built-in loaders.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def load_mnist():
    """
    Load MNIST handwritten digits (0-9).
    Returns:
        (x_train, y_train), (x_test, y_test) as numpy arrays.
        Images are normalized to [0,1] and reshaped to (28,28,1).
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values from [0,255] to [0,1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension for CNN (28,28,1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    print(f"MNIST loaded: {x_train.shape[0]} training, {x_test.shape[0]} test samples")
    return (x_train, y_train), (x_test, y_test)

def load_emnist_letters():
    """
    Load EMNIST Letters dataset (A-Z, 26 classes).
    Note: EMNIST has several splits. We use 'letters' which contains uppercase/lowercase merged.
    Labels are 1-26 (A=1, B=2, ... Z=26). We shift to 0-25.
    """
    # EMNIST letters are rotated and flipped. We need to rotate back.
    (x_train, y_train), (x_test, y_test) = keras.datasets.emnist.load_data(type='letters')
    
    # EMNIST images are transposed (flipped horizontally and rotated 90 deg)
    # We fix by rotating 90 deg clockwise and flipping horizontally
    x_train = np.rot90(x_train, k=-1, axes=(1,2))
    x_test = np.rot90(x_test, k=-1, axes=(1,2))
    x_train = np.flip(x_train, axis=2)
    x_test = np.flip(x_test, axis=2)
    
    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    # Shift labels from 1-26 to 0-25
    y_train = y_train - 1
    y_test = y_test - 1
    
    print(f"EMNIST Letters loaded: {x_train.shape[0]} training, {x_test.shape[0]} test samples")
    return (x_train, y_train), (x_test, y_test)

def preview_samples(images, labels, class_names, num_samples=10):
    """
    Display a grid of sample images with their labels.
    """
    plt.figure(figsize=(10, 4))
    for i in range(num_samples):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Label: {class_names[labels[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage (run as script for testing)
if __name__ == "__main__":
    # Test MNIST
    (x_train_mnist, y_train_mnist), _ = load_mnist()
    class_names_mnist = [str(i) for i in range(10)]
    preview_samples(x_train_mnist, y_train_mnist, class_names_mnist)
    
    # Test EMNIST
    (x_train_emnist, y_train_emnist), _ = load_emnist_letters()
    class_names_emnist = [chr(ord('A')+i) for i in range(26)]
    preview_samples(x_train_emnist, y_train_emnist, class_names_emnist)
