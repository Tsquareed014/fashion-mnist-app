import streamlit as st
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
from io import BytesIO

# Load the trained model
model = tf.keras.models.load_model("fashion_mnist_cnn_model.keras")

# Define class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to preprocess image
def preprocess_image(image):
    image = image.convert("L").resize((28, 28))  # Convert to grayscale & resize
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Fashion MNIST Image Classifier 👕👗👞")

# Upload Image
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and Predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    # Display Result
    st.write(f"### Predicted Class: {predicted_class}")
    st.write(f"### Confidence: {confidence:.2f}")

