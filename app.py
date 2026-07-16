import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# 1. Website interface settings
st.title("🧠 AI Brain Tumor Detector")
st.write("Upload an MRI image so the system can analyze it and show the result instantly.")

# 2. Load the "brain" of the trained model
# Used cache so the model loads only once and the website doesn't become slow
@st.cache_resource 
def load_model():
    return tf.keras.models.load_model('brain_tumor_detector.h5')

model = load_model()

# 3. File upload button for the client
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

# If the client uploaded an image, execute the following:
if uploaded_file is not None:
    # Display the image on the website
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_container_width=True)
    
    st.write("Performing detailed analysis... ⏳")
    
    # 4. Prepare the image the same way the model was trained on
    img_array = np.array(image.convert('L')) # Convert to grayscale
    img_resized = cv2.resize(img_array, (64, 64)) # Resize to 64x64
    img_reshaped = img_resized.reshape(-1, 64, 64, 1) / 255.0 # Normalize pixel values
    
    # 5. The moment of truth (analysis)
    prediction = model.predict(img_reshaped)
    
    st.write("---")
    # Display the result in a nice way
    if prediction[0][0] > 0.5:
        st.error("🚨 Result: Tumor Detected")
    else:
        st.success("✅ Result: Healthy")