"""
Main entry point - demonstrates end-to-end workflow.
Run this script to:
1. Train a model (MNIST by default)
2. Evaluate on test set
3. Optionally predict on a sample image
"""

import argparse
from src.train import train_model
from src.evaluate import evaluate_model
import os

def main():
    parser = argparse.ArgumentParser(description='Handwritten Character Recognition Pipeline')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'predict'],
                        help='Operation mode')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'emnist'], help='Dataset')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model for evaluation/prediction')
    parser.add_argument('--image', type=str, default=None,
                        help='Image path for prediction')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Train new model
        train_model(dataset=args.dataset, epochs=10, batch_size=128, save_dir='models')
        
    elif args.mode == 'evaluate':
        if not args.model_path:
            # Try default path
            args.model_path = f'models/{args.dataset}_model.h5'
            if not os.path.exists(args.model_path):
                print(f"Default model not found at {args.model_path}. Please specify --model_path.")
                return
        evaluate_model(args.model_path, dataset=args.dataset)
        
    elif args.mode == 'predict':
        if not args.model_path:
            args.model_path = f'models/{args.dataset}_model.h5'
        if not os.path.exists(args.model_path):
            print(f"Model not found at {args.model_path}")
            return
        
        # Import predict module dynamically to avoid circular import
        from src.predict import predict_image, draw_and_predict
        from tensorflow import keras
        
        model = keras.models.load_model(args.model_path)
        if args.dataset == 'mnist':
            class_names = [str(i) for i in range(10)]
        else:
            class_names = [chr(ord('A')+i) for i in range(26)]
        
        if args.image:
            predict_image(model, args.image, class_names)
        else:
            draw_and_predict(model, class_names)

if __name__ == "__main__":
    main()
