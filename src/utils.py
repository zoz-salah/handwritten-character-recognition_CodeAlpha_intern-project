"""
Utility functions for visualization and model saving/loading.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def display_predictions(model, images, true_labels, class_names, num_images=10):
    """
    Display a grid of images with true and predicted labels.
    """
    predictions = model.predict(images[:num_images])
    pred_labels = np.argmax(predictions, axis=1)
    
    plt.figure(figsize=(12, 5))
    for i in range(num_images):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        color = 'green' if pred_labels[i] == true_labels[i] else 'red'
        plt.title(f"True: {class_names[true_labels[i]]}\nPred: {class_names[pred_labels[i]]}", 
                  color=color, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_model_summary(model, filepath):
    """Save model architecture summary to text file."""
    with open(filepath, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
