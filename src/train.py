# src/train.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import yaml
import argparse
from model import advanced_unet_model

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Training Script for Semantic Segmentation')
parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
parser.add_argument('--epochs', type=int, default=None, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
args = parser.parse_args()

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

config = load_config(args.config)

# Override config parameters with command-line arguments if provided
if args.epochs is not None:
    config['training']['epochs'] = args.epochs

if args.batch_size is not None:
    config['training']['batch_size'] = args.batch_size

# Load data
images = np.load('data/images.npy')
masks = np.load('data/masks.npy')

# Split data
train_images, val_images, train_masks, val_masks = train_test_split(
    images, masks, 
    test_size=1-config['dataset']['train_split'], 
    random_state=42
)

# Define ImageDataGenerators for data augmentation
data_gen_args = dict(
    rotation_range=config['augmentation']['rotation_range'],
    width_shift_range=config['augmentation']['width_shift_range'],
    height_shift_range=config['augmentation']['height_shift_range'],
    shear_range=config['augmentation']['shear_range'],
    zoom_range=config['augmentation']['zoom_range'],
    horizontal_flip=config['augmentation']['horizontal_flip'],
    vertical_flip=config['augmentation']['vertical_flip'],
    fill_mode=config['augmentation']['fill_mode']
)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Create data generators
train_image_generator = image_datagen.flow(train_images, batch_size=config['training']['batch_size'], seed=42)
train_mask_generator = mask_datagen.flow(train_masks, batch_size=config['training']['batch_size'], seed=42)
train_generator = zip(train_image_generator, train_mask_generator)

val_image_generator = image_datagen.flow(val_images, batch_size=config['training']['batch_size'], seed=42)
val_mask_generator = mask_datagen.flow(val_masks, batch_size=config['training']['batch_size'], seed=42)
val_generator = zip(val_image_generator, val_mask_generator)

# Build and compile model
model = advanced_unet_model(
    input_size=tuple(config['model']['input_size']), 
    num_classes=config['model']['num_classes'],
    dropout=config['model']['dropout']
)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
callbacks = [
    EarlyStopping(patience=config['training']['patience'], verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(patience=config['training']['reduce_lr_patience'], verbose=1),
    ModelCheckpoint(config['paths']['best_model'], save_best_only=True, verbose=1)
]

# Train model
model.fit(
    train_generator,
    steps_per_epoch=len(train_images) // config['training']['batch_size'],
    epochs=config['training']['epochs'],
    validation_data=val_generator,
    validation_steps=len(val_images) // config['training']['batch_size'],
    callbacks=callbacks
)

# Save trained model
model.save(config['paths']['best_model'])
