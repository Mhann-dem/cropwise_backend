import io
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
from ultralytics import YOLO
import logging
import os

# Initialize Flask app
app = Flask(__name__)

# CORS setup - restrict origins in production
CORS(app, resources={r"/*": {"origins": "*"}})  # Change "*" to specific domains in production

# Setup logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Load YOLOv8 model
model = YOLO("trained_30.pt")  # Ensure the path is correct

@app.route("/", methods=["GET"])
def say_hello():
    return jsonify("Hello and welcome to plant disease classification!")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception as e:
        return jsonify({'error': 'Invalid image file'}), 400

    try:
        # Perform the prediction
        results = model(image)

        # Handle model results
        if results and len(results) > 0:
            result = results[0]  # Assuming we need the first result
            
            # Get the prediction with the highest confidence
            pred = result.probs.top1  # Index of the highest probability
            conf = result.probs.top1conf.item()  # Confidence of the highest probability
            
            top_prediction = {
                'label': result.names[pred],
                'confidence': float(conf)
            }
        else:
            return jsonify({'error': 'No valid prediction made'}), 500

        return jsonify({'prediction': top_prediction}), 200

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': 'An error occurred during prediction. Please try again later.'}), 500
    
if __name__ == '__main__':
    # Use a production-grade WSGI server like Gunicorn when deploying
    app.run(host='0.0.0.0', port=5000)
