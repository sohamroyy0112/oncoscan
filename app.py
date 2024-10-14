import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model  # Import load_model
from PIL import Image
import numpy as np
import zipfile
import os

# Unzipping the model file (only if it's zipped)
if not os.path.exists('leukemia_classification_model_optimized.h5'):
    with zipfile.ZipFile('leukemia_classification_model_optimized.zip', 'r') as zip_ref:
        zip_ref.extractall()

# Load the unzipped model
model = load_model('leukemia_classification_model_optimized.h5')

# Function to preprocess images
def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app interface
st.title("Leukemia Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make a prediction
    prediction = model.predict(preprocessed_image)

    # Display the result
    result = 'ALL' if prediction[0][0] > 0.5 else 'HEM'
    st.write(f"Prediction: {result}")
