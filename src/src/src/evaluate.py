"""
Evaluate trained model on test dataset and generate classification report.
"""

import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from data_loader import load_mnist, load_emnist_letters

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

def evaluate_model(model_path, dataset='mnist'):
    """
    Load a saved model and evaluate on test set.
    Prints accuracy and classification report.
    """
    # Load model
    model = keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    # Load test data
    if dataset.lower() == 'mnist':
        (_, _), (x_test, y_test) = load_mnist()
        class_names = [str(i) for i in range(10)]
    elif dataset.lower() == 'emnist':
        (_, _), (x_test, y_test) = load_emnist_letters()
        class_names = [chr(ord('A')+i) for i in range(26)]
    else:
        raise ValueError("Dataset must be 'mnist' or 'emnist'")
    
    # Evaluate overall accuracy
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    
    # Get predictions
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, class_names, 
                          save_path=f'{dataset}_confusion_matrix.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained model.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved .h5 model file')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'emnist'], help='Dataset used for training')
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.dataset)
