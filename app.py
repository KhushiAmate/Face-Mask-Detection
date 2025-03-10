import os
import cv2
import numpy as np
import urllib.request
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model

# Flask app configuration
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('models/face_mask_model.keras')

# Check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to render HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        prediction = predict_mask(file_path)
        os.remove(file_path)  # Clean up uploaded file
        return jsonify({'prediction': prediction})
    return jsonify({'error': 'Invalid file type'})

# Route to handle URL-based predictions
@app.route('/predict_url', methods=['POST'])
def predict_url():
    image_url = request.form['url']
    try:
        response = urllib.request.urlopen(image_url)
        image_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        prediction = predict_mask_from_image(image)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': 'Unable to process the image URL'})

# Function to predict mask from an image path
def predict_mask(image_path):
    image = cv2.imread(image_path)
    return predict_mask_from_image(image)

# Function to predict mask from a raw image
def predict_mask_from_image(image):
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype="float32") / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    mask_label = 'Mask' if predictions[0][0] > 0.5 else 'No Mask'
    return mask_label

# Main function to run the Flask app
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
