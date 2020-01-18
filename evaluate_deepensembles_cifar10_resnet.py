# Partially based on Keras' CIFAR10 resnet example
# Available at: https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py

import keras_uncertainty
from keras_uncertainty.utils import numpy_negative_log_likelihood

import keras
import keras.backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.layers import Activation, AveragePooling2D
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

from ResNet import resnet_layer

def train_cnn(x_train, y_train):
    input_shape = (32, 32, 3)
    # Start model definition.
    num_filters = 16
    num_res_blocks = 3

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)

    # Stack 0, Residual block 0
    y = resnet_layer(inputs=x, num_filters=num_filters, strides=1)
    y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
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

    num_filters *= 2

    # Stack 1, Residual block 0
    y = resnet_layer(inputs=x, num_filters=num_filters, strides=2)
    y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
    x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=2, activation=None, batch_normalization=False)
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)

    # Stack 1, Residual block 1
    y = resnet_layer(inputs=x, num_filters=num_filters, strides=1)
    y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)

    # Stack 1, Residual block 2
    y = resnet_layer(inputs=x, num_filters=num_filters, strides=1)
    y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)

    num_filters *= 2

    # Stack 2, Residual block 0
    y = resnet_layer(inputs=x, num_filters=num_filters, strides=2)
    y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
    x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=2, activation=None, batch_normalization=False)
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)

    # Stack 2, Residual block 1
    y = resnet_layer(inputs=x, num_filters=num_filters, strides=1)
    y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)

    # Stack 2, Residual block 2
    y = resnet_layer(inputs=x, num_filters=num_filters, strides=1)
    y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(10, activation='softmax', kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)

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

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=128), epochs=100, verbose=2)
    
    return model

# Experiment hyperparams

NUM_ENSEMBLES = list(range(1,16))

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

    results = pd.DataFrame(columns = ['num_ensembles', 'error', 'nll'])

    ensemble_models = []

    for num_ens in NUM_ENSEMBLES:

        #Add one member to the ensemble    
        ensemble_models.append(train_cnn(x_train, y_train))

        if num_ens == 1:
            ensemble_models[0].summary()

        #Evaluate ensemble
        preds = []

        for model in ensemble_models:
            preds.append(np.expand_dims(model.predict(x_test, verbose=0), axis=0))

        preds = np.concatenate(preds, axis=0)

        mean_pred = np.mean(preds, axis=0)
        mean_pred = mean_pred / np.sum(mean_pred, axis=1, keepdims=True)
        class_true = np.argmax(y_test, axis=1)
        class_pred = np.argmax(mean_pred, axis=1)

        acc = accuracy_score(class_true, class_pred)
        err = 100.0 * (1.0 - acc)
        nll = numpy_negative_log_likelihood(y_test, mean_pred)

        print("{} Ensembles, Error: {:.4f} NLL {:.4f}".format(num_ens, err, nll))

        results = results.append({'num_ensembles': num_ens, 'error': err, 'nll': nll}, ignore_index=True)

    results.to_csv('deepensembles_resnet_cifar10.csv', sep=';', index=False)
