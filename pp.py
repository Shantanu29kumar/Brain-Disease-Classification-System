# import cv2
# import numpy as np
# from tensorflow import keras
# from cv2 import imshow
# # Assuming x_test_CNN is already processed and reshaped
# model = keras.models.load_model('trained_model.h5')
# image_CNN = cv2.resize(cv2.imread('bd937738ad6223a03f8aedcf4920a7_thumb.jpeg',cv2.IMREAD_GRAYSCALE),(150,150))
# x_test_CNN = np.array(image_CNN)/255.0
# image_to_display = (x_test_CNN * 255).astype(np.uint8)  # Convert back to uint8
# cv2.imshow('Edited Image', image_to_display)
# cv2.waitKey(0)  # Wait for a key press to close the window
# cv2.destroyAllWindows()

from tensorflow import keras
import cv2
import numpy as np

# Load the trained model
model = keras.models.load_model('trained_model.h5')

# Load and preprocess the single image (replace 'your_image.jpg' with the actual image file)
image_CNN = cv2.resize(cv2.imread('preopnonfunctionalmacroadenoma2.jpg', cv2.IMREAD_GRAYSCALE), (150, 150))
x_test_CNN = np.array(image_CNN) / 255.0
x_test_CNN = x_test_CNN.reshape(-1, 150, 150, 1)

# Make a prediction
predictions = model.predict(x_test_CNN)

# Define class labels
class_labels = ['glioma', 'meningioma', 'MildDemented', 'ModerateDemented', 'NonDemented', 'notumor', 'pituitary', 'VeryMildDemented']

# Print the predicted class and its probability
predicted_class = class_labels[np.argmax(predictions)]
probability = predictions[0][np.argmax(predictions)]

print(f'Predicted Class: {predicted_class}')
print(f'Probability: {probability:.2%}')
