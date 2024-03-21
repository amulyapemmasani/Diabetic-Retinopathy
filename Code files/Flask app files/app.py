from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
app = Flask(__name__)
 
upload_folder = os.path.join('static', 'uploads')
 
app.config['UPLOAD'] = upload_folder

# Set the maximum content length to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load the model
# The model file is in the '/home/ubuntu' directory
model = load_model('tensorflow_model.h5')
class_labels = ['0', '1', '2', '3', '4']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print(request.files)

    if 'image' not in request.files:
            return "No file part", 400
    image_file = request.files['image']
    if image_file.filename == '':
        return "No selected file", 400
    if image_file:
        image_file = request.files['image']
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (300, 300))
        
        # Preprocess the image as required by your model
        X = np.array(img).reshape(1, 300, 300, 3)
        predictions = model.predict(X)
        print(predictions)
        predicted_label = class_labels[np.argmax(predictions[0])]
        print(predicted_label)
        # Map the predicted label to a human-readable text
        label_mapping = {
            '0': "No DR",
            '1': "Mild",
            '2': "Moderate",
            '3': "Severe",
            '4': "Proliferative DR"
        }
        text = label_mapping.get(predicted_label, "No DR")
        
        return render_template('result.html', text=text)

    return 'No image provided', 400

if __name__ == '__main__':
    app.run(port=8000, debug=True)

