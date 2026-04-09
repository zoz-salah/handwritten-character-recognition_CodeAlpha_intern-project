"""
Training script for the CNN model on MNIST or EMNIST dataset.
Saves the trained model and training history plots.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from data_loader import load_mnist, load_emnist_letters
from model import create_cnn_model, compile_model

def plot_training_history(history, save_path=None):
    """
    Plot training & validation accuracy and loss.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    plt.show()

def train_model(dataset='mnist', epochs=10, batch_size=128, save_dir='models'):
    """
    Train the CNN on selected dataset.
    
    Args:
        dataset: 'mnist' or 'emnist'
        epochs: number of training epochs
        batch_size: batch size
        save_dir: directory to save model and plots
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    if dataset.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = load_mnist()
        num_classes = 10
        class_names = [str(i) for i in range(10)]
    elif dataset.lower() == 'emnist':
        (x_train, y_train), (x_test, y_test) = load_emnist_letters()
        num_classes = 26
        class_names = [chr(ord('A')+i) for i in range(26)]
    else:
        raise ValueError("Dataset must be 'mnist' or 'emnist'")
    
    # Create validation split from training data (10% for validation)
    val_split = 0.1
    val_size = int(len(x_train) * val_split)
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]
    
    print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")
    
    # Build and compile model
    model = create_cnn_model(input_shape=(28, 28, 1), num_classes=num_classes)
    model = compile_model(model)
    
    # Optional: Add callbacks (early stopping, model checkpoint)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, f'{dataset}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(save_dir, f'{dataset}_model.h5')
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Plot history
    plot_path = os.path.join(save_dir, f'{dataset}_training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train handwritten character recognition model.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'emnist'],
                        help='Dataset to use (mnist or emnist)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save model')
    
    args = parser.parse_args()
    
    train_model(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
