'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import backend as K
from keras.utils.vis_utils import plot_model
import numpy as np
#  import matplotlib
#  matplotlib.use('Agg')
#  import matplotlib.pyplot as plt

#  batch_size = 128
batch_size = 32
num_classes = 25
epochs = 30

# input image dimensions
img_rows, img_cols = 28, 28

## LOAD DATA ##
#  data_dir = '/mnt/c/Users/Scott/classes/b657/project/data/small/'
data_dir = '../data/small/'
train = np.loadtxt(open(data_dir + 'sign_mnist_train.csv'), delimiter=',', skiprows=1)
train_augmented = np.loadtxt(open(data_dir + 'train_augmented.csv'), delimiter=',')
x_train = train[:,1:] 
x_train_augmented = train_augmented[:,1:]
y_train = train[:,0]
y_train_augmented = train_augmented[:,0]

x_train = np.vstack((x_train, x_train_augmented))
y_train = np.concatenate((y_train, y_train_augmented))
#  x_train = x_train_augmented
#  y_train = y_train_augmented

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
test = np.loadtxt(open(data_dir + 'sign_mnist_test.csv'), delimiter=',', skiprows=1)
x_test = test[:,1:]
y_test = test[:,0]
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

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
#  plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(Conv2D(28, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(28, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(28, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(56, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(28, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

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

model.save('../models/asl_model_alt_2.h5')

#  plt.plot(history.history['acc'])
#  plt.plot(history.history['val_acc'])
#  plt.title('Model Accuracy')
#  plt.ylabel('Accuracy')
#  plt.xlabel('Epoch')
#  plt.legend(['train', 'test'], loc='upper left')
#  plt.savefig('accuracy')
#  plt.clf()
#  # summarize history for loss
#  plt.plot(history.history['loss'])
#  plt.plot(history.history['val_loss'])
#  plt.title('model loss')
#  plt.ylabel('loss')
#  plt.xlabel('epoch')
#  plt.legend(['train', 'test'], loc='upper left')
#  plt.show()
#  plt.savefig('loss')
