from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model("fruits_recognition_model.h5")
IMG_SIZE = 100
CATEGORIES = ['Blueberry', 'Pear', 'Strawberry', 'Avocado', 'Pomegranate', 'Pepper Red', 'Kiwi', 'Lemon', 'Raspberry', 'Plum', 'Cherry', 'Cucumber Ripe', 'Clementine', 'Watermelon', 'Cantaloupe', 'Apple Braeburn', 'Onion White', 'Mango', 'Potato Red', 'Passion Fruit', 'Apple Granny Smith', 'Apricot', 'Limes', 'Corn', 'Banana', 'Grape Blue', 'Cactus fruit', 'Papaya', 'Pineapple', 'Tomato', 'Orange', 'Pepper Green', 'Peach']  # Replace with your dataset categories

# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        img = Image.open(file).resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)
        prediction = np.argmax(model.predict(img_array), axis=1)[0]
        return jsonify({'predicted_class': CATEGORIES[prediction]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check route
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
