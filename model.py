
#
# Model Training script for CarND Project 3
#
#
#
#

# Load Python Modules
import pickle
import numpy as np
import tensorflow as tf
import sklearn
import os
import csv
import cv2
import matplotlib.pyplot as plt

# Load Keras Modules
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

# Read entries from CSV Files
# samples are preprocessed, sample[0] = full filename, sample[1] = angle
samples = []

# 1. Training data provided by Udacity
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        try:
            center_angle = float(line[3])
            name = 'data/IMG/'+line[0].split('/')[-1]
            
            newline = []
            newline.append(name)
            newline.append(center_angle)
            
            # left camera
            leftname = 'data/IMG/'+line[1].split('/')[-1]
            leftangle = center_angle + 0.35
            leftline = []
            leftline.append(leftname)
            leftline.append(leftangle)
            samples.append(leftline)
            
            # right camera
            rightname = 'data/IMG/'+line[2].split('/')[-1]
            rightangle = center_angle - 0.35
            rightline = []
            rightline.append(rightname)
            rightline.append(rightangle)
            samples.append(rightline)
            
            samples.append(newline)
            
            # more emphasis on large steering samples
            if center_angle < -0.15:
                samples.append(newline)
                samples.append(newline)
            if center_angle > 0.15:
                samples.append(newline)
                samples.append(newline)
        except ValueError:
            a = 1

# 2. Training data recorded: after bridge            
with open('turnafterbridge/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        try:
            center_angle = float(line[3])
            name = 'turnafterbridge/IMG/'+line[0].split('/')[-1]
            
            newline = []
            newline.append(name)
            newline.append(center_angle)

            # left camera
            leftname = 'turnafterbridge/IMG/'+line[1].split('/')[-1]
            leftangle = center_angle + 0.35
            leftline = []
            leftline.append(leftname)
            leftline.append(leftangle)
            samples.append(leftline)
            
            # right camera
            rightname = 'turnafterbridge/IMG/'+line[2].split('/')[-1]
            rightangle = center_angle - 0.35
            rightline = []
            rightline.append(rightname)
            rightline.append(rightangle)
            samples.append(rightline)
            
            samples.append(newline)
            samples.append(newline)
            samples.append(newline)
            # more emphasis on large steering samples
            if center_angle < -0.15:
                samples.append(newline)
                samples.append(newline)
                samples.append(newline)
                samples.append(newline)
                samples.append(newline)
                samples.append(newline)
                samples.append(newline)
                samples.append(newline)
                samples.append(newline)
            if center_angle > 0.15:
                samples.append(newline)
                samples.append(newline)
        except ValueError:
            a = 1

# 3. Training data recorded: two turns after sharp turn after bridge            
with open('twoturns/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        try:
            center_angle = float(line[3])
            name = 'twoturns/IMG/'+line[0].split('/')[-1]
            
            newline = []
            newline.append(name)
            newline.append(center_angle)

            # left camera
            leftname = 'twoturns/IMG/'+line[1].split('/')[-1]
            leftangle = center_angle + 0.45
            leftline = []
            leftline.append(leftname)
            leftline.append(leftangle)
            samples.append(leftline)
            
            # right camera
            rightname = 'twoturns/IMG/'+line[2].split('/')[-1]
            rightangle = center_angle - 0.45
            rightline = []
            rightline.append(rightname)
            rightline.append(rightangle)
            samples.append(rightline)
            
            samples.append(newline)
            samples.append(newline)
            samples.append(newline)
            # more emphasis on large steering samples
            if center_angle < -0.15:
                samples.append(newline)
                samples.append(newline)
                samples.append(newline)
                samples.append(newline)
                samples.append(newline)
                samples.append(newline)
                samples.append(newline)
                samples.append(newline)
                samples.append(newline)
            if center_angle > 0.15:
                samples.append(newline)
                samples.append(newline)
        except ValueError:
            a = 1

# split data set into training and validation            
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print("Train Sample Size",len(train_samples))
print("Validation Sample Size",len(validation_samples))

# samples are preprocessed, sample[0] = full filename, sample[1] = angle
# generator is used so that we don't need to load all images into memory
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                try:
                    center_angle = float(batch_sample[1])
                    name = batch_sample[0]
                    srcBGR = cv2.imread(name)
                    center_image = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
                    images.append(center_image)
                    angles.append(center_angle)
                except ValueError:
                    a = 1
                    print("Not a number in ",batch_sample[0]," Value: ",batch_sample[1])
                except Exception:
                    print("Image error ",batch_sample[0]," Value: ",batch_sample[1])
                    
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
batch_size = 32

train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

ch, row, col = 3, 160, 320  # Trimmed image format

# Model is based on Nvidia model
model = Sequential()
# crop out the top and bottom parts of the image
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(row, col, ch)))
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x/127.5) - 1.0))
# Use Max Pooling to reduce image size by half
model.add(MaxPooling2D((2, 2)))
# 5 convolution layers
model.add(Convolution2D(24, (5, 5), strides=2))
model.add(Convolution2D(36, (5, 5), strides=2))
model.add(Convolution2D(48, (5, 5), strides=1))
model.add(Convolution2D(64, (3, 3), strides=1))
model.add(Convolution2D(64, (3, 3), strides=1))
model.add(Flatten())
# 5 Fully Connected Layers, without 1 Dropout layer in the middle
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Dense(10))
#model.add(Activation('relu'))
model.add(Dense(1))

# use Adam optimizer
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/batch_size -1, 
                    validation_data=validation_generator, 
                    validation_steps=len(validation_samples)/batch_size -1, 
                    epochs=5)

modelfile = 'car-model-13.h5'
model.save(modelfile)

print("Model saved.",modelfile)


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
