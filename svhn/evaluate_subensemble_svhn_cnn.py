import keras_uncertainty
from keras_uncertainty.utils import numpy_negative_log_likelihood

import keras, sys
import keras.backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.models import Model, Sequential
from keras.datasets import mnist

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

from svhn_io import load_svhn

def cnn_module(inp, num_filters, num_modules = 2):
    x = inp

    for i in range(num_modules):
        x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x) 

    x = MaxPooling2D((2,2))(x)

    return x

CNN_LAYERS = [
    lambda x: cnn_module(x, 32),
    lambda x: cnn_module(x, 64),
    lambda x: cnn_module(x, 128),
    lambda x: cnn_module(x, 128, num_modules=3),
    lambda x: BatchNormalization()(Dense(128, activation='relu')(Flatten()(x)))
]

def train_cnn(x_train, y_train, base_layers, subensemble_layers, return_base=True):
    inp = Input(shape=(32, 32, 3))
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
    inp = Input(shape=(32, 32, 3))

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
MAX_LAYERS = len(CNN_LAYERS)
SUBENSEMBLE_LAYERS = 2

if __name__ == "__main__":
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = load_svhn()

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')    

    from subensembles import evaluate_subensemble_layers
    from keras.datasets import cifar10
    
    (_, __), (x_ood, ___) = cifar10.load_data()

    results = evaluate_subensemble_layers("cnn_svhn-sub-{}-".format(SUBENSEMBLE_LAYERS), NUM_ENSEMBLES, x_train, y_train, x_test, y_test, x_ood, train_cnn, train_sub_cnn, MAX_LAYERS, SUBENSEMBLE_LAYERS)
    results.to_csv('sub-deepensembles_{}_cnn_svhn.csv'.format(SUBENSEMBLE_LAYERS), sep=';', index=False)

    sys.exit(0)

    # Multiple runs correlation experiment

    RUNS = 10
    
    for run in range(RUNS):
        results = evaluate_subensemble_layers("cnn_svhn-sub-{}-".format(SUBENSEMBLE_LAYERS), NUM_ENSEMBLES, x_train, y_train, x_test, y_test, x_ood, train_cnn, train_sub_cnn, MAX_LAYERS, SUBENSEMBLE_LAYERS)
        
        base_error = results['error'][0]
        base_nll = results['nll'][0]

        results.insert(len(results.columns), 'base_error', base_error)
        results.insert(len(results.columns), 'base_nll', base_nll)

        results.to_csv('sub-deepensembles_{}_cnn_svhn-run{}.csv'.format(SUBENSEMBLE_LAYERS, run), sep=';', index=False)
