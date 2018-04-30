'''
Trains the hand detection model.

This file modified (significantly) from here:
    https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import backend as K
#  from keras.utils.vis_utils import plot_model
import numpy as np
import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

batch_size = 20
#  batch_size = 1
num_classes = 2
epochs = 30

# input image dimensions
img_rows, img_cols = 100, 100

## LOAD DATA ##
#  data_dir = '/mnt/c/Users/Scott/classes/b657/project/data/small/'
data_dir = '../data/localization/'

print('Loading data')
data = pandas.read_csv(open(data_dir + 'localization_100.csv'), nrows=2000)
#  data = pandas.read_csv(open(data_dir + 'localization_28.csv'))
train = data.values
#  train = np.loadtxt(open(data_dir + 'localization_100.csv'), delimiter=',', skiprows=0)

print('Making Views')
x = train[:, 1:]
y = train[:, 0]

print(x.shape)
print(y.shape)

x_train = x[0:1000,:]
y_train = y[0:1000]
x_test = x[1000:1500,:]
y_test = y[1000:1500]

print('Reshaping')
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#  test = np.loadtxt(open(data_dir + 'sign_mnist_test.csv'), delimiter=',', skiprows=1)
#  x_test = test[:,1:]
#  y_test = test[:,0]
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

print('Normalizing')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# ORIGINAL MODEL
#  model = Sequential()
#  model.add(Conv2D(32, kernel_size=(5, 5),
                     #  activation='relu',
                     #  input_shape=input_shape))
#  model.add(Conv2D(64, (3, 3), activation='relu'))
#  model.add(MaxPooling2D(pool_size=(2, 2)))
#  model.add(Conv2D(32, (3, 3), activation='relu'))
#  model.add(MaxPooling2D(pool_size=(2, 2)))
#  model.add(Dropout(0.25))
#  model.add(Flatten())
#  model.add(Dense(128, activation='relu'))
#  model.add(Dense(98, activation='relu'))
#  model.add(Dense(52, activation='relu'))
#  model.add(Dropout(0.5))
#  model.add(Dense(num_classes, activation='softmax'))

model = Sequential()
model.add(Conv2D(100, kernel_size=(5,5), activation='relu', input_shape=input_shape))
model.add(Conv2D(50, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
#  model.add(Conv2D(30, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(30, kernel_size=(3,3), activation='relu'))
#  model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

#  plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(loss=keras.losses.categorical_crossentropy,
                      #  optimizer=keras.optimizers.Adadelta(),
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('../models/localization_model_2.h5')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy')
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('loss')
