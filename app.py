from flask import Flask, request
import os
import cv2
import numpy as np
from tensorflow import keras

app = Flask(__name__)

# Define a function to make predictions
def predict_disease(file_path):
    # Load the trained model
    model = keras.models.load_model('trained_model.h5')
    
    # Check if the file exists and is an allowed file type (e.g., image)
    if not file_path or not allowed_file(file_path):
        return "Invalid file. Please upload an image."

    # Load and preprocess the image
    image_CNN = cv2.resize(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE), (150, 150))
    x_test_CNN = np.array(image_CNN) / 255.0
    x_test_CNN = x_test_CNN.reshape(-1, 150, 150, 1)

    # Make a prediction
    predictions = model.predict(x_test_CNN)

    # Define class labels
    class_labels = ['Glioma', 'Meningioma', 'Mild Demented', 'Moderate Demented', 'NonDemented', 'No Tumor', 'Pituitary', 'Very Mild Demented']

    # Get the predicted class and its probability
    predicted_class = class_labels[np.argmax(predictions)]
    probability = predictions[0][np.argmax(predictions)]

    return f'Predicted Class: {predicted_class}, Probability: {probability:.2f}'

# Check if the uploaded file is allowed (you can extend this list)
def allowed_file(filename):
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Route for handling file uploads and predictions
@app.route('/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part."

    file = request.files['file']
    if file.filename == '':
        return "No selected file."

    # Save the uploaded file temporarily (you may want to handle this differently)
    file_path = 'temp_image.jpg'
    file.save(file_path)

    # Make predictions using the uploaded image
    prediction_result = predict_disease(file_path)

    # Remove the temporary file
    os.remove(file_path)

    return prediction_result

if __name__ == '__main__':
    app.run(debug=True)
