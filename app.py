from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import gdown

app = Flask(__name__)

# Path where the model will be downloaded temporarily
MODEL_PATH = 'models/model.h5'

# Function to download the model if it doesn't exist
def download_model():
    if not os.path.exists('models'):
        os.makedirs('models')  # Create the 'models' directory if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        # Use your Google Drive file ID
        gdown.download('https://drive.google.com/uc?id=12br5OceyvFex-ktOjGF1_hXId8yk2Ok0', MODEL_PATH, quiet=False)
        print("Model downloaded successfully.")

# Download the model at the start of the app
download_model()

# Load the model
model = load_model(MODEL_PATH)

IMG_SIZE = (69, 69)  # Example image size, you may need to adjust it

@app.route('/')
def home():
    return "Flask App is Running"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file:
        try:
            # Save the uploaded image temporarily
            filepath = os.path.join('/tmp', file.filename)
            file.save(filepath)

            # Preprocess the image
            img = load_img(filepath, target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            img_array = img_array.reshape(1, *img_array.shape)

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = int(prediction.argmax())

            return jsonify({
                'predicted_class': predicted_class,
                'confidence': prediction[0][predicted_class]
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File upload failed'}), 400

if __name__ == "__main__":
    app.run(debug=True)
