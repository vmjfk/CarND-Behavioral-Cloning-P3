#Keras model for self drivng car experiment
#Udacity self driving car engineer nano degree
#

# Import the data
import csv
import cv2
import numpy as np 
path = '.\\data\\'
driving_log = []
with open(path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if 'steering' not in line: 
            driving_log.append(line)

images = []
measurements = []

print("Loading images from disk...")
for i in range(len(driving_log)):
    line = driving_log[i]
    fn = line[0].split('/')[-1]
    fn = fn.split('\\')[-1]
    fn = path + "IMG\\" + fn
    image = cv2.imread(fn)
    nimage = np.array(image)
    images.append(image)
    measurements.append(float(line[3]))
print("Images loaded from disk.")


X_train = np.array(images)
y_train = np.array(measurements)
print("X_train shape= ",X_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D,Cropping2D


X_train = np.array(images)
y_train = np.array(measurements)
print("X_train shape= ",X_train.shape)

print("Building the model.")
model = Sequential()
model.add(Lambda(lambda x: x /255.0 -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=2)


print("Saving model.h5")
model.save('model.h5')

