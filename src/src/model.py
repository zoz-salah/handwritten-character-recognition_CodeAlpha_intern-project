"""
CNN model architecture for handwritten character recognition.
The same architecture works for both MNIST (10 classes) and EMNIST (26 classes).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Build a simple but effective Convolutional Neural Network.
    
    Architecture:
        Conv2D(32, 3x3) -> ReLU -> MaxPool2D(2x2)
        Conv2D(64, 3x3) -> ReLU -> MaxPool2D(2x2)
        Conv2D(128, 3x3) -> ReLU
        Flatten
        Dense(128) -> ReLU -> Dropout(0.5)
        Dense(num_classes) -> Softmax
    
    Args:
        input_shape: tuple (height, width, channels)
        num_classes: number of output classes (10 for digits, 26 for letters)
    Returns:
        Compiled Keras model.
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block (no pooling)
        layers.Conv2D(128, (3, 3), activation='relu'),
        
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Helps prevent overfitting
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile the model with Adam optimizer and sparse categorical crossentropy.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Test model creation
if __name__ == "__main__":
    # For MNIST
    model_mnist = create_cnn_model(num_classes=10)
    model_mnist = compile_model(model_mnist)
    model_mnist.summary()
    
    # For EMNIST Letters
    model_emnist = create_cnn_model(num_classes=26)
    model_emnist = compile_model(model_emnist)
    model_emnist.summary()
