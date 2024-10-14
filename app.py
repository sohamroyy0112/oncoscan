import zipfile
import os


if not os.path.exists('leukemia_classification_model_optimized.h5'):
    with zipfile.ZipFile('leukemia_classification_model_optimized.zip', 'r') as zip_ref:
        zip_ref.extractall()


model = load_model('leukemia_classification_model_optimized.h5')



import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


model = load_model('leukemia_classification_model_optimized.h5')


def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image)
    image = image / 255.0  
    image = np.expand_dims(image, axis=0) 
    return image


st.title("Leukemia Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    

    preprocessed_image = preprocess_image(image)


    prediction = model.predict(preprocessed_image)

    # Display result
    result = 'ALL' if prediction[0][0] > 0.5 else 'HEM'
    st.write(f"Prediction: {result}")
