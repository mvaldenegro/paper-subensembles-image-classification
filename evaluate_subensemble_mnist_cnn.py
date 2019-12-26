# Partially based on Keras' MNIST CNN example
# Available at: https://raw.githubusercontent.com/keras-team/keras/master/examples/mnist_cnn.py

import keras_uncertainty
from keras_uncertainty.models import DeepSubensembleRegressor, DeepSubensembleClassifier
from keras_uncertainty.utils import numpy_negative_log_likelihood

import keras
import keras.backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.models import Model, Sequential
from keras.datasets import mnist

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

CNN_LAYERS = [
    lambda x: BatchNormalization()(Conv2D(32, kernel_size=(3, 3), activation='relu')(x)),
    lambda x: BatchNormalization()(Conv2D(64, kernel_size=(3, 3), activation='relu')(x)),
    lambda x: Dense(128, activation='relu')(Flatten()(MaxPooling2D((2,2))(x)))
]

def train_cnn(x_train, y_train, base_layers, subensemble_layers, return_base=True):
    inp = Input(shape=(28, 28, 1))
    x = inp

    for i in range(base_layers):
        x = CNN_LAYERS[i](x)

    base_model = Model(inp, x)

    for i in range(base_layers, base_layers + subensemble_layers):
        x = CNN_LAYERS[i](x)

    out = Dense(10, activation="softmax")(x)
    model = Model(inp, out)

    model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=30, verbose=0)
    
    if return_base:
        return model, base_model

    return model

def train_sub_cnn(base_model, x_train, y_train, base_layers, subensemble_layers):
    inp = Input(shape=(28, 28, 1))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model(inp)

    for i in range(base_layers, base_layers + subensemble_layers):
        x = CNN_LAYERS[i](x)

    out = Dense(10, activation="softmax")(x)
    model = Model(inp, out)

    model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=30, verbose=0)

    return model

# Experiment hyperparams

NUM_ENSEMBLES = list(range(1,16))
MAX_LAYERS = 3
SUBENSEMBLE_LAYERS = 1

if __name__ == "__main__":
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1, img_rows, img_cols, 1))
    x_test = x_test.reshape((-1, img_rows, img_cols, 1))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')    

    from subensembles import evaluate_subensemble_layers

    #results = evaluate_subensemble_layers(NUM_ENSEMBLES, x_train, y_train, x_test, y_test, train_cnn, train_sub_cnn, MAX_LAYERS, SUBENSEMBLE_LAYERS)
    #results.to_csv('sub-deepensembles_{}_cnn_mnist.csv'.format(SUBENSEMBLE_LAYERS), sep=';', index=False)

    # Multiple runs correlation experiment

    RUNS = 10
    
    for run in range(RUNS):
        results = evaluate_subensemble_layers(NUM_ENSEMBLES, x_train, y_train, x_test, y_test, train_cnn, train_sub_cnn, MAX_LAYERS, SUBENSEMBLE_LAYERS)
        
        base_error = results['error'][0]
        base_nll = results['nll'][0]

        results.insert(len(results.columns), 'base_error', base_error)
        results.insert(len(results.columns), 'base_nll', base_nll)

        results.to_csv('sub-deepensembles_{}_cnn_mnist-run{}.csv'.format(SUBENSEMBLE_LAYERS, run), sep=';', index=False)
