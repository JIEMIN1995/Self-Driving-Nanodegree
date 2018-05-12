import csv
import cv2
import numpy as np

lines = []
with open('data/driving_log_win.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
idx = 0
for line in lines:                                                                        
    idx += 1
    if idx == 1:
        continue
  
    steering_center = float(line[3])
    
    # create adjusted steering measurements for the side camera images
    correction = 0.2
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    # read in images from center, left and right cameras
    path = "data/IMG_win/"
    filename_center = line[0].split('/')[-1]
    filename_left = line[1].split('/')[-1]
    filename_right = line[2].split('/')[-1]
    img_center = cv2.imread(path + filename_center)
    img_left = cv2.imread(path + filename_left)
    img_right = cv2.imread(path + filename_right)
    
    # add images and angles to data set
    images.append(img_center)
    images.append(img_left)
    images.append(img_right)
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)
    
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)    
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(BatchNormalization())
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(BatchNormalization())
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(BatchNormalization())
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(BatchNormalization())
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, 
          y_train, 
          batch_size=30,
          validation_split=0.2,
          shuffle=True, 
          epochs=3)
model.save('model.h5')

