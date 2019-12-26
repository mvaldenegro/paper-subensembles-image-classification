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

def train_cnn(x_train, y_train):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=30, verbose=0)
    
    return model

# Experiment hyperparams

NUM_ENSEMBLES = list(range(1,16))

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

    results = pd.DataFrame(columns = ['num_ensembles', 'error', 'nll'])

    ensemble_models = []

    for num_ens in NUM_ENSEMBLES:

        #Add one member to the ensemble    
        ensemble_models.append(train_cnn(x_train, y_train))
            
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

    results.to_csv('deepensembles_cnn_mnist.csv', sep=';', index=False)