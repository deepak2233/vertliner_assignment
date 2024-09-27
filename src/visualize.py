# src/visualize.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import yaml
import argparse

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Visualization Script for Semantic Segmentation')
parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
parser.add_argument('--model_path', type=str, default=None, help='Path to the trained model file')
args = parser.parse_args()

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

config = load_config(args.config)

# Override config parameters with command-line arguments if provided
if args.model_path is not None:
    config['paths']['best_model'] = args.model_path

# Load data
val_images = np.load('data/val_images.npy')
val_masks = np.load('data/val_masks.npy')

# Load model
model = load_model(config['paths']['best_model'])

# Evaluate model
val_loss, val_acc = model.evaluate(val_images, val_masks)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

# Predict on validation data
val_predictions = model.predict(val_images)
val_pred_class_indices = np.argmax(val_predictions, axis=-1)

# Save segmentation results
output_dir = config['paths']['predictions']
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(len(val_images)):
    result_image = val_images[i]
    result_mask = val_pred_class_indices[i]

    plt.imsave(os.path.join(output_dir, f"result_image_{i}.png"), result_image)
    plt.imsave(os.path.join(output_dir, f"result_mask_{i}.png"), result_mask, cmap='tab20')

print("Segmentation results saved successfully!")
