import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import gdown

# --- Load model from Google Drive ---
MODEL_PATH = "model.h5"
FILE_ID = "1kIDT8hr8N62gGveHhA8ch_GuxVZT-Nww"

if not os.path.exists(MODEL_PATH):
    st.write("🔄 Downloading model...")
    url = f"https://drive.google.com/uc?id=1kIDT8hr8N62gGveHhA8ch_GuxVZT-Nww"
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

# --- Streamlit UI ---
st.title("🐶 Dog vs 🐱 Cat Classifier")

uploaded_file = st.file_uploader("Upload an image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((256, 256))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prob = model.predict(img_array)[0][0]
    pred_class = "Dog 🐶" if prob > 0.5 else "Cat 🐱"

    st.write(f"### ✅ Prediction: **{pred_class}**")
    st.write(f"Confidence: `{prob:.2f}`")
