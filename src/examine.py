''' let's examine the training data cuz i don't understand the labels on it. '''

import numpy as np
import cv2

data_dir = '../data/small/'
print('loading data...')
train = np.loadtxt(open(data_dir + 'sign_mnist_train.csv'), delimiter=',', skiprows=1)

x_train = train[:,1:]
x_train = x_train.reshape(x_train.shape[0], 28, 28)
y_train = train[:,0]

for i in range(x_train.shape[0]):
    if y_train[i] != 9.0:
        continue
    # normalize pixels
    img = x_train[i]
    for rowi, row in enumerate(img):
        for coli, col in enumerate(row):
            img[rowi][coli] = img[rowi][coli] / 255.
    print(img)
    print(y_train[i])
    while True:
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print()
            break
