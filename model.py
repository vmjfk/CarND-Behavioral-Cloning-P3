#Keras model for self drivng car experiment
#Udacity self driving car engineer nano degree
#

# Import the data
import csv
import cv2


driving_log = []
with open('./data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if 'steering' not in line: 
            driving_log.append(line)

images = []
measurements = []

print("Loading images from disk...")
for i in range(len(driving_log)):
    line = driving_log[i]
    fn = line[1].split('/')[-1]
    path = './data/data/IMG/' + fn
    images.append(cv2.imread(path))
    measurements.append(float(line[3]))
print("Images loaded from disk.")

import numpy as np
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D

print("Building the model.")
model = Sequential()
model.add(Lambda(lambda x: x /255.0 -0.5, input_shape=(160,320,3)))

model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(84))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=2)


print("Saving model.h5")
model.save('model.h5')

