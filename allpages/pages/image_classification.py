import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model(r"C:\Users\Gamer Tech\Desktop\FYP\models\potato_disease_model.h5")

# Define class labels
class_labels = ["Early Blight", "Healthy", "Late Blight"]

disease_info = {
    "Early Blight": "Early blight is a common fungal disease in potatoes caused by *Alternaria solani*. It causes dark brown spots on leaves, leading to reduced yield. Prevention includes crop rotation, proper spacing, and fungicide application.",
    "Healthy": "The potato leaf appears healthy with no visible signs of disease. Maintain proper watering, nutrient levels, and pest control to keep plants healthy.",
    "Late Blight": "Late blight is a serious potato disease caused by *Phytophthora infestans*. It spreads rapidly in wet conditions, causing dark lesions on leaves and stems. It was responsible for the Irish Potato Famine. Management includes fungicides and resistant varieties."
}



# Image classification page
st.title("🖼️ Potato Leaf Disease Classification")
st.write("Upload a potato leaf image to predict the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((256, 256))  # Resize to model input size
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    predicted_label = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display result
    st.write(f"### 🏆 Prediction: **{predicted_label}**")
    st.write(f"### 🔥 Confidence: **{confidence:.2f}%**")
    st.info(disease_info[predicted_label])

# Back to Main Page Button

