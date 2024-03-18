from tensorflow import keras
import cv2
import numpy as np
from cv2 import imshow

model = keras.models.load_model('trained_model.h5')
image_CNN = cv2.resize(cv2.imread('bd937738ad6223a03f8aedcf4920a7_thumb.jpeg',cv2.IMREAD_GRAYSCALE),(150,150))
x_test_CNN = np.array(image_CNN)/255.0
print(x_test_CNN.shape)
# x_test_CNN  = x_test_CNN[:,:,np.newaxis]
# prediction = model.predict(x_test_CNN)
# print(prediction)

x_test_CNN = x_test_CNN.reshape(-1, 150, 150, 1)
x_test_CNN = x_test_CNN / 255.0
predictions = model.predict(x_test_CNN)
print(predictions)

# imshow(x_test_CNN)