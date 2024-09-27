# src/data_loader.py

import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
import yaml
import argparse

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Data Loader for Semantic Segmentation')
parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
args = parser.parse_args()

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

config = load_config(args.config)

# Define class mapping
color_to_class = {
    (226, 169, 41): 0,  # Water
    (132, 41, 246): 1,  # Land
    (110, 193, 228): 2, # Road
    (60, 16, 152): 3,   # Building
    (254, 221, 58): 4,  # Vegetation
    (155, 155, 155): 5  # Unlabeled
}

def count_class_pixels(mask_array, num_classes):
    counts = np.zeros(num_classes, dtype=int)
    for rgb, class_idx in color_to_class.items():
        match = np.all(mask_array == rgb, axis=-1)
        counts[class_idx] += np.sum(match)
    return counts

def load_images_and_masks(dataset_folder_path, image_size=(256, 256)):
    images, masks = []
    for tile in os.listdir(dataset_folder_path):
        tile_path = os.path.join(dataset_folder_path, tile)
        if os.path.isdir(tile_path):
            images_folder = os.path.join(tile_path, 'images')
            masks_folder = os.path.join(tile_path, 'masks')
            for image_file in os.listdir(images_folder):
                img_path = os.path.join(images_folder, image_file)
                mask_file = image_file.replace('.jpg', '.png')
                mask_path = os.path.join(masks_folder, mask_file)
                if os.path.exists(mask_path):
                    image = Image.open(img_path).resize(image_size)
                    mask = Image.open(mask_path).resize(image_size).convert('RGB')
                    images.append(np.array(image) / 255.0)
                    masks.append(np.array(mask))
    return np.array(images), np.array(masks)

def preprocess_masks(masks, num_classes=6):
    processed_masks = []
    for mask in masks:
        mask_class_indices = np.zeros(mask.shape[:2], dtype=int)
        for rgb, class_idx in color_to_class.items():
            match = np.all(mask == rgb, axis=-1)
            mask_class_indices[match] = class_idx
        mask_one_hot = to_categorical(mask_class_indices, num_classes=num_classes)
        processed_masks.append(mask_one_hot)
    return np.array(processed_masks)

if __name__ == '__main__':
    images, masks = load_images_and_masks(config['dataset']['path'], tuple(config['dataset']['image_size']))
    masks = preprocess_masks(masks, config['dataset']['num_classes'])
    np.save('data/images.npy', images)
    np.save('data/masks.npy', masks)
