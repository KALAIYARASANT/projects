import streamlit as st
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model("tumor_model.keras")
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

st.title("Brain Tumor MRI Image Classifier")
st.write("Upload an MRI image to predict the tumor type.")

uploaded_file = st.file_uploader("Choose an image..", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', width=400)
    img = np.array(img)
    img = cv2.resize(img,(150,150))
    img = img.reshape(1,150,150,3)

    prediction = model.predict(img)
    predicted_label = labels[np.argmax(prediction)]

    st.success(f"Prediction: **{predicted_label}**")
