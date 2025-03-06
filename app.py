from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the pre-trained model (ensure the model file is in your project directory)
model = tf.keras.models.load_model("fashion_mnist_cnn_model.keras")

# Define class names corresponding to model output indices
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess_image(image):
    # Convert image to grayscale and resize to (28,28)
    image = image.convert("L").resize((28, 28))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Expecting a JSON payload with a base64-encoded image (e.g., {"image": "base64string"})
    image_data = data.get("image")
    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
    except Exception as e:
        return jsonify({"error": "Invalid image data", "details": str(e)}), 400

    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    
    return jsonify({"predicted_class": predicted_class, "confidence": confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)


