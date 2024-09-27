# streamlit_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import yaml
from PIL import Image
import argparse

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Streamlit App for Semantic Segmentation Inference')
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

model = load_model(config['paths']['best_model'])

# Streamlit UI
st.title("Semantic Segmentation Inference")
st.write("Upload an image to perform semantic segmentation.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_resized = image.resize(tuple(config['model']['input_size'][:2]))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Perform prediction
    prediction = model.predict(image_array)
    predicted_mask = np.argmax(prediction, axis=-1).squeeze()

    # Display input image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Display predicted mask
    st.write("Predicted Mask:")
    plt.figure(figsize=(10, 10))
    plt.imshow(predicted_mask, cmap='tab20')
    st.pyplot(plt)
