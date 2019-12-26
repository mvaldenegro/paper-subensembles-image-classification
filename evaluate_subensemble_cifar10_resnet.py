# Partially based on Keras' CIFAR10 resnet example
# Available at: https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py

import keras_uncertainty
from keras_uncertainty.models import DeepSubensembleRegressor, DeepSubensembleClassifier
from keras_uncertainty.utils import numpy_negative_log_likelihood

import keras, sys
import keras.backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.layers import Activation, AveragePooling2D
from keras.models import Model, Sequential
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

from ResNet import resnet_layer

def resnet_stack(inp, num_filters, strides=1):
    x = inp

    # Stack 0, Residual block 0
    y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
    y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)

    if strides > 1:
        x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None, batch_normalization=False)
        
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)

    # Stack 0, Residual block 1
    y = resnet_layer(inputs=x, num_filters=num_filters, strides=1)
    y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)

    # Stack 0, Residual block 2
    y = resnet_layer(inputs=x, num_filters=num_filters, strides=1)
    y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)

    return x

def resnet_classifier(x):
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(10, activation='softmax', kernel_initializer='he_normal')(y)

    return outputs

CNN_LAYERS = [
    lambda x: resnet_layer(inputs=x),
    lambda x: resnet_stack(x, 16),
    lambda x: resnet_stack(x, 32, strides=2),
    lambda x: resnet_stack(x, 64, strides=2)
]

def train_cnn(x_train, y_train, base_layers, subensemble_layers, return_base=True):
    inp = Input(shape=(32, 32, 3))
    x = inp

    for i in range(base_layers):
        x = CNN_LAYERS[i](x)

    base_model = Model(inp, x)

    for i in range(base_layers, base_layers + subensemble_layers):
        x = CNN_LAYERS[i](x)

    out = resnet_classifier(x)
    model = Model(inp, out)

    model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    datagen.fit(x_train)

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=128), epochs=100, verbose=2)
    
    if return_base:
        return model, base_model

    return model

def train_sub_cnn(base_model, x_train, y_train, base_layers, subensemble_layers):
    inp = Input(shape=(32, 32, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model(inp)

    for i in range(base_layers, base_layers + subensemble_layers):
        x = CNN_LAYERS[i](x)

    out = resnet_classifier(x)
    model = Model(inp, out)

    model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    datagen.fit(x_train)

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=128), epochs=100, verbose=2)

    return model

# Experiment hyperparams

NUM_ENSEMBLES = list(range(1,16))
MAX_LAYERS = len(CNN_LAYERS)
SUBENSEMBLE_LAYERS = 1

if __name__ == "__main__":
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')    

    from subensembles import evaluate_subensemble_layers

    results = evaluate_subensemble_layers(NUM_ENSEMBLES, x_train, y_train, x_test, y_test, train_cnn, train_sub_cnn, MAX_LAYERS, SUBENSEMBLE_LAYERS)
    results.to_csv('sub-deepensembles_{}_cnn_svhn.csv'.format(SUBENSEMBLE_LAYERS), sep=';', index=False)

    sys.exit(0)

    # Multiple runs correlation experiment

    RUNS = 10
    
    for run in range(RUNS):
        results = evaluate_subensemble_layers(NUM_ENSEMBLES, x_train, y_train, x_test, y_test, train_cnn, train_sub_cnn, MAX_LAYERS, SUBENSEMBLE_LAYERS)
        
        base_error = results['error'][0]
        base_nll = results['nll'][0]

        results.insert(len(results.columns), 'base_error', base_error)
        results.insert(len(results.columns), 'base_nll', base_nll)

        results.to_csv('sub-deepensembles_{}_cnn_svhn-run{}.csv'.format(SUBENSEMBLE_LAYERS, run), sep=';', index=False)
