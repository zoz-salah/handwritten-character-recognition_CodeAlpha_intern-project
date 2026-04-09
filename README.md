# Handwritten Character Recognition with CNN

A beginner-friendly deep learning project for recognizing handwritten digits (MNIST) and letters (EMNIST) using Convolutional Neural Networks.

## Features

- Train CNN on MNIST (digits) or EMNIST (letters A-Z)
- Evaluate model with accuracy, classification report, and confusion matrix
- Predict custom images or draw characters interactively
- Clean, well-commented code for learning

## Project Structure
handwritten-character-recognition/
├── data/ # Auto-downloaded datasets
├── models/ # Saved trained models
├── src/ # Source code
│ ├── data_loader.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│ ├── predict.py
│ └── utils.py
├── main.py # Entry point
├── requirements.txt
└── README.md

text

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/handwritten-character-recognition.git
   cd handwritten-character-recognition
Create a virtual environment (recommended):

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
pip install -r requirements.txt
Usage
Train a Model
bash
# Train on MNIST digits
python main.py --mode train --dataset mnist

# Train on EMNIST letters
python main.py --mode train --dataset emnist --epochs 15
Trained models will be saved in models/.

Evaluate on Test Set
bash
python main.py --mode evaluate --dataset mnist --model_path models/mnist_model.h5
This shows accuracy, classification report, and a confusion matrix.

Predict Custom Images
bash
# Predict from an image file
python main.py --mode predict --dataset mnist --image path/to/digit.png

# Interactive drawing canvas (no --image argument)
python main.py --mode predict --dataset mnist
Extending to Word/Sentence Recognition
This project can be extended to full word recognition using:

Segmentation: Detect and isolate individual characters in a word image.

Sliding Window + CNN: Use the trained CNN to classify each segment.

Sequence Models: Combine CNN with RNN/LSTM or CTC loss for end-to-end recognition.

For a simple approach, you can use contour detection (OpenCV) to segment characters, then pass each to the prediction function.
