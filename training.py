import numpy as np
import matplotlib.pyplot as plt

import cv2

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# This script trains a simple convolutional neural network (CNN)
# for the MNIST dataset. The code comes from one of the examples
# provided in the documentation for keras.

batch_size = 128
num_classes = 10
epochs = 1

# Input image dimensions
img_rows, img_cols = 14, 14

# MNIST data, split between training and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# To save time, further reduce the size of the training dataset.
training_data_length = 20000
x_train = x_train[:training_data_length]
y_train = y_train[:training_data_length]

# Each image is resized from (28, 28) to (14, 14). 
x_train = np.array([cv2.resize(image, (img_rows, img_cols)) for image in x_train])
x_test = np.array([cv2.resize(image, (img_rows, img_cols)) for image in x_test])

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation=None, name='logit'))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the trained model, including the weights as well as
# the configuration. 
model.save('MNIST_simple_model.h5')

sample = x_train[0,:,:,0]
plt.imshow(sample, cmap='gray')
plt.show()
prediction = model.predict(sample.reshape((1, img_rows, img_cols, 1)))
print("The prediction result is ", prediction)
