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
    print(fn)
    image = cv2.imread(fn)
    nimage = np.array(image)
    print(nimage.shape)
    images.append(image)
    measurements.append(float(line[3]))
print("Images loaded from disk.")


X_train = np.array(images)
y_train = np.array(measurements)
print("X_train shape= ",X_train.shape)


