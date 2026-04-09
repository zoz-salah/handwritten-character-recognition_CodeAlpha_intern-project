"""
Predict handwritten characters from custom images.
Supports single image file or drawing using OpenCV canvas.
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def preprocess_image(image_path, invert=True):
    """
    Load image, convert to grayscale, resize to 28x28, and normalize.
    Assumes white background with dark character. Invert if needed.
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Resize to 28x28 (model input size)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Invert if needed: model expects black background, white digit (like MNIST)
    # Most user images are dark on light, so invert.
    if invert:
        img = 255 - img
    
    # Normalize to [0,1]
    img = img.astype('float32') / 255.0
    
    # Add batch and channel dimensions: (1, 28, 28, 1)
    img = np.expand_dims(img, axis=(0, -1))
    
    return img

def predict_image(model, image_path, class_names, invert=True):
    """
    Predict class for a single image.
    """
    img = preprocess_image(image_path, invert)
    
    # Get prediction
    probs = model.predict(img, verbose=0)[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    
    # Show image and prediction
    plt.imshow(img[0, :, :, 0], cmap='gray')
    plt.title(f"Predicted: {class_names[pred_idx]} ({confidence:.2%})")
    plt.axis('off')
    plt.show()
    
    print(f"Predicted class: {class_names[pred_idx]}")
    print(f"Confidence: {confidence:.4f}")
    print("All probabilities:")
    for i, prob in enumerate(probs):
        print(f"  {class_names[i]}: {prob:.4f}")
    
    return pred_idx, confidence

def draw_and_predict(model, class_names):
    """
    Open a drawing canvas with OpenCV. User can draw a character and press 'p' to predict,
    'c' to clear, 'q' to quit.
    """
    canvas = np.ones((280, 280), dtype=np.uint8) * 255  # white background
    drawing = False
    
    def draw(event, x, y, flags, param):
        nonlocal drawing, canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(canvas, (x, y), 15, (0, 0, 0), -1)  # black brush
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
    
    cv2.namedWindow('Draw a character')
    cv2.setMouseCallback('Draw a character', draw)
    
    print("Draw a character using mouse. Press:")
    print("  'p' - predict")
    print("  'c' - clear canvas")
    print("  'q' - quit")
    
    while True:
        cv2.imshow('Draw a character', canvas)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('p'):
            # Preprocess canvas image
            img = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
            # Invert: black on white -> white on black
            img = 255 - img
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=(0, -1))
            
            # Predict
            probs = model.predict(img, verbose=0)[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            
            print(f"Predicted: {class_names[pred_idx]} (confidence: {confidence:.2%})")
            
            # Show processed image
            plt.imshow(img[0, :, :, 0], cmap='gray')
            plt.title(f"Prediction: {class_names[pred_idx]}")
            plt.axis('off')
            plt.show()
            
        elif key == ord('c'):
            canvas[:] = 255
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict handwritten character from image or drawing.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained .h5 model')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to image file (if not provided, opens drawing canvas)')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'emnist'], help='Model type (mnist=digits, emnist=letters)')
    
    args = parser.parse_args()
    
    # Load model
    model = keras.models.load_model(args.model_path)
    print(f"Model loaded from {args.model_path}")
    
    # Define class names based on dataset
    if args.dataset == 'mnist':
        class_names = [str(i) for i in range(10)]
    else:  # emnist
        class_names = [chr(ord('A')+i) for i in range(26)]
    
    if args.image:
        predict_image(model, args.image, class_names, invert=True)
    else:
        draw_and_predict(model, class_names)
