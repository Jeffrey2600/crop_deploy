from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load your CNN model from the local path
model = load_model('models/model.h5')  # Ensure correct path to model file
IMG_SIZE = (69, 69)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file:
        try:
            # Save the file temporarily
            filepath = os.path.join('/tmp', file.filename)
            file.save(filepath)

            # Preprocess the image
            img = load_img(filepath, target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make predictions
            prediction = model.predict(img_array)
            os.remove(filepath)

            # Get the predicted class and confidence
            predicted_class_idx = int(np.argmax(prediction, axis=1)[0])
            confidence = float(prediction[0][predicted_class_idx])

            class_labels = ['Cercospora_Leaf_Spot', 'Common_Rust', 'Healthy', 'Northern_Leaf_Blight']
            predicted_label = class_labels[predicted_class_idx]

            return jsonify({
                'predicted_class': predicted_label,
                'confidence': confidence
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File upload failed'}), 400

if __name__ == "__main__":
    app.run()
