import streamlit as st
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model('model.h5')

# UI
st.title("🐶🐱 Dog vs Cat Classifier")

uploaded_file = st.file_uploader("Upload a dog or cat image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((256, 256))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prob = model.predict(img_array)[0][0]
        pred_class = "Dog 🐶" if prob > 0.5 else "Cat 🐱"

        st.write(f"### Prediction: {pred_class}")
        st.write(f"Confidence: {prob:.2f}")
