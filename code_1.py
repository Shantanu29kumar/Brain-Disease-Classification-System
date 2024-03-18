import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random

train_path = 'combined_dataset/training'
test_path = 'combined_dataset/testing'

def y_2binary(n):
    z=np.zeros((8), dtype=int)
    z[n]=1
    return z


### Get a list of subdirectories of the train_path
sub_dirs = os.listdir(train_path)  


### We shall map names of the subdirectories to numerical values 0, 1, ..., using a dictionary.
output_dictionary = {}   
for i in range(len(sub_dirs)):
    output_dictionary.update({sub_dirs[i]: i})

print('The subdirectories are',sub_dirs,'.\n')


### Create empty lists for training/testing data, since we don't know how big they will be.
x_train_CNN = []  
x_test_CNN = []  

x_train_MobileNetV2 = []
x_test_MobileNetV2 = []

### The training/testing output data is the same for both models. 
y_train = []
y_test = []

### Calculate the number of training samples in each subdirectory, find the total number of training samples. 
### Convert images to size (150,150) for CNN model, and to size (224,224,3) for MobilenetV2 model. Convert 
### the lists to arrays, and store array in x_train. 

for dir in sub_dirs:
    sub_dir_path_train = train_path + '/' + dir
    file_counter_train=0
    for path in os.listdir(sub_dir_path_train):
        x=os.path.join(sub_dir_path_train,path)
        if os.path.isfile(x):
            file_counter_train+=1
            
            image_CNN = cv2.resize(cv2.imread(x,cv2.IMREAD_GRAYSCALE),(150,150))
            image_MobileNetV2 = cv2.resize(cv2.imread(x),(224,224))
            
            x_train_CNN.append(image_CNN)
            x_train_MobileNetV2.append(image_MobileNetV2)
            y_train.append(y_2binary(output_dictionary[dir]))
            
    print('The training directory',dir,'has',file_counter_train,'samples.')
print()

### Calculate the number of testing samples in each subdirectory, find the total number of testing samples. 
### Convert images to size (224,224,3), then convert them to an np.array, and store array in x_test.

for dir in sub_dirs:
    sub_dir_path_test = test_path + '/' + dir
    file_counter_test=0
    for path in os.listdir(sub_dir_path_test):
        x=os.path.join(sub_dir_path_test,path)
        if os.path.isfile(x):
            file_counter_test+=1
            
            image_CNN = cv2.resize(cv2.imread(x,cv2.IMREAD_GRAYSCALE),(150,150))
            #image_MobileNetV2 = cv2.resize(cv2.imread(x),(224,224))
            
            x_test_CNN.append(image_CNN)
            #x_test_MobileNetV2.append(image_MobileNetV2)
            y_test.append(y_2binary(output_dictionary[dir]))
            
    print('The testing directory',dir,'has',file_counter_test,'samples.')
print()

### Convert the now fully populated lists to numpy arrays. 
x_train_CNN = np.array(x_train_CNN)/255.0
x_test_CNN = np.array(x_test_CNN)/255.0
x_train_CNN  = x_train_CNN[:,:,:,np.newaxis]
x_test_CNN  =  x_test_CNN[:,:,:,np.newaxis] 

x_train_MobileNetV2 = np.array(x_train_MobileNetV2)/255.0
x_test_MobileNetV2 = np.array(x_test_MobileNetV2)/255.0

y_train = np.array(y_train)
y_test = np.array(y_test)


### Print information summary, and a sample training and test image. 
print()
print('The shape of x_train_CNN is',x_train_CNN.shape,'.')
print('The shape of x_test_CNN is',x_test_CNN.shape,'.\n')
print('The shape of x_train_MobileNetV2 is',x_train_MobileNetV2.shape,'.')
print('The shape of x_test_MobileNetV2 is',x_test_MobileNetV2.shape,'.\n')
print('The shape of y_train is',y_train.shape,'.')
print('The shape of y_test is',y_test.shape,'.\n')


print('The total number of training samples is',len(y_train),'.')
print('The total number of testing samples is',len(y_test),'.\n')


n1 = random.randint(0,len(y_train))
plt.imshow(x_train_CNN[n1], cmap='gray')
plt.title('Sample Trainig Scan')
plt.show()

print() 

n2 = random.randint(0,len(x_test_CNN))
plt.imshow(x_test_CNN[n2], cmap='gray')
plt.title('Sample Testing Scan')
plt.show()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input, Dense, InputLayer, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Model, Sequential
from keras import metrics

model_CNN = Sequential()
model_CNN.add(InputLayer(input_shape=(150,150,1)))

model_CNN.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model_CNN.add(MaxPooling2D(pool_size=(2,2)))
model_CNN.add(Dropout(0.25))

model_CNN.add(Conv2D(filters=64, kernel_size=(4,4), activation='relu'))
model_CNN.add(MaxPooling2D(pool_size=(2,2)))
model_CNN.add(Dropout(0.25))

model_CNN.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model_CNN.add(MaxPooling2D(pool_size=(2,2)))
model_CNN.add(Dropout(0.25))

model_CNN.add(Conv2D(filters=256, kernel_size=(2,2), activation='relu'))
model_CNN.add(MaxPooling2D(pool_size=(2,2)))
model_CNN.add(Dropout(0.25))

model_CNN.add(Flatten())
model_CNN.add(Dense(2048, activation='relu'))
model_CNN.add(Dropout(0.25))

model_CNN.add(Dense(8, activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99)
model_CNN.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model_CNN.summary()

import time
import math

t1=time.time()
history_CNN = model_CNN.fit(x_train_CNN, y_train, batch_size = 64, epochs = 30)
model_CNN.save('CNN_30epochs')
t2=time.time()
print('The training of 30 epochs of the CNN model took',round((t2-t1)/60),'minutes.')

model_CNN.save('trained_model.h5')

