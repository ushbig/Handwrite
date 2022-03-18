import tensorflow as tf
import keras as keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import utils  
import numpy as np
from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import ImageOps
from IPython.display import display
model = load_model('mnist.h5')

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_test.shape, y_test.shape)

print( y_train.shape)


##Result(10000, 28, 28) (10000,)
##(60000,)


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes=None)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=None)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape, 'test samples')
print(y_test.shape, 'y_test samples')
print(y_train.dtype, y_train.shape[0])

#x_train shape: (60000, 28, 28, 1)
#60000 train samples
#(10000, 28, 28, 1) test samples
#(10000, 10) y_test samples
#float32 60000

batch_size = 128
num_classes = 10
epochs = 10

model = keras.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer=tf.keras.optimizers.Adadelta(),metrics=['accuracy'])


hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=0,validation_data=(x_test, y_test))
print("The model has successfully trained")

#model.save('mnist.h5')
print("Saving the model as mnist.h5")

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
