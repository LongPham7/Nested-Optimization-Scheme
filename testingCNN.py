import numpy as np
import matplotlib.pyplot as plt

import cv2

import keras
from keras.datasets import mnist
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from optimization_problem import OptimizationProblem
from sequential_method_multidimensional import MultidimensionalSequentialMethod

batch_size = 128
num_classes = 10
epochs = 2

# Input image dimensions
img_rows, img_cols = 14, 14

# MNIST data, split between training and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

training_data_length = 20000

x_train = x_train[:training_data_length]
y_train = y_train[:training_data_length]

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

# Load the trained model.
model = load_model('MNIST_simple_model.h5')

logit_model = Model(inputs=model.input, outputs=model.get_layer('logit').output)

sample = x_train[0,:,:,0]
plt.imshow(sample, cmap='gray')
plt.show()

result = model.predict(sample.reshape([1, img_rows, img_cols, 1]))
print("The correct result is", y_train[0])
print("The predicted softmax result is ", result)

def modifyPixels(image, pixels, *newValues):
    """
    Modify pixels at specified coordinates.
    """
    newImage = np.copy(image)
    newValuesList = [v for v in newValues]

    # The number of coordinates should be equal
    # to the number of arguments supplied to this
    # method. 
    assert len(pixels) == len(newValuesList)

    for i in range(len(pixels)):
        (x, y) = pixels[i]
        newImage[x, y] = newValuesList[i]
    return newImage

def predictLabelForSingleImage(model, image, label):
    newImage = image.reshape((1, img_rows, img_cols, 1))
    return model.predict(newImage)[0, label]

# Coordinates of the pixels to be modified
coordinates = [(3,7), (3,8), (4,7), (4,8), (5,7), (5,8), (6,7), (6,8)]

# For sanity check, modify the specified pixels and display the resulting image.
newImage = modifyPixels(sample, coordinates, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
plt.imshow(newImage, cmap='gray')
plt.show()

# Prediction on the modified image
softmax_result = predictLabelForSingleImage(model, newImage, 5) # The correct label is 5.
print("The softmax result of prediction is", softmax_result)
logit_result = predictLabelForSingleImage(logit_model, newImage, 5) # The correct label is 5.
print("The logit result of prediction is", logit_result)


# Multidimensional function whose global extrema are to be computed
target = lambda *xs: predictLabelForSingleImage(logit_model, modifyPixels(sample, coordinates, *xs), 5)
domain = [(0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1)]
problem = OptimizationProblem(target, domain)
optimizer = MultidimensionalSequentialMethod(problem)
minimum = optimizer.computeMinimum(0.05)
print("The minimum is", minimum)
#maximum = optimizer.computeMaximum(0.001)
#print("The maxium is", maximum)
