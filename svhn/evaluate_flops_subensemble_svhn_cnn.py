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

def build_model(base_layers, subensemble_layers, return_base=True):
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
    
    if return_base:
        return model, base_model

    return model

# Experiment hyperparams

NUM_ENSEMBLES = list(range(1,16))
MAX_LAYERS = len(CNN_LAYERS)
SUBENSEMBLE_LAYERS = 4

from count_flops import count_model_params_flops

if __name__ == "__main__":
    model, base_model = build_model(MAX_LAYERS - SUBENSEMBLE_LAYERS, SUBENSEMBLE_LAYERS)

    _, model_flops = count_model_params_flops(model)
    _, base_model_flops = count_model_params_flops(base_model)
    subensemble_flops = model_flops - base_model_flops

    print("Model FLOPS: {}".format(model_flops))
    print("Base model FLOPS: {}".format(base_model_flops))
    print("Per-Subensemble FLOPS: {}".format(subensemble_flops))
    
    results = pd.DataFrame(columns = ['num_ensembles', 'speedup'])

    for i in NUM_ENSEMBLES:
        total_flops_full = i * model_flops 
        total_flops_sub = base_model_flops + i * subensemble_flops
        speedup = total_flops_full / total_flops_sub

        print("{} -> Deep Ensemble FLOPS: {}, Sub-Ensemble FLOPS: {} Speedup: {:.2f}".format(i, total_flops_full, total_flops_sub, speedup))

        results = results.append({'num_ensembles': i, 'speedup': speedup}, ignore_index=True)

    results.to_csv('sub-deepensembles_svhn_cnn_speedup-{}.csv'.format(SUBENSEMBLE_LAYERS), sep=';', index=False)