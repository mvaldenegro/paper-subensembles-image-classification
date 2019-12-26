import keras_uncertainty
from keras_uncertainty.utils import numpy_negative_log_likelihood

import keras
import keras.backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.models import Model, Sequential
from keras.datasets import mnist

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from svhn_io import load_svhn
from subensembles import save_predictions, ensemble_predictions

def train_cnn(x_train, y_train):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=30, verbose=0)
    
    return model

# Experiment hyperparams

NUM_ENSEMBLES = list(range(1,16))
EXPERIMENT_NAME = "deepensembles-cnn_svhn"

if __name__ == "__main__":
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = load_svhn()

    from keras.datasets import cifar10
    
    (_, __), (x_ood, ___) = cifar10.load_data()

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
        mean_pred = ensemble_predictions(ensemble_models, x_test)
        class_true = np.argmax(y_test, axis=1)
        class_pred = np.argmax(mean_pred, axis=1)

        save_predictions("{}-num_ens-{}-preds.hdf5".format(EXPERIMENT_NAME, num_ens), mean_pred)

        #Evaluate ensembe on OOD data
        ood_preds = ensemble_predictions(ensemble_models, x_ood)
        save_predictions("{}-num_ens-{}-ood-preds.hdf5".format(EXPERIMENT_NAME, num_ens), ood_preds)

        acc = accuracy_score(class_true, class_pred)
        err = 100.0 * (1.0 - acc)
        nll = numpy_negative_log_likelihood(y_test, mean_pred)

        print("{} Ensembles, Error: {:.4f} NLL {:.4f}".format(num_ens, err, nll))

        results = results.append({'num_ensembles': num_ens, 'error': err, 'nll': nll}, ignore_index=True)

    results.to_csv('deepensembles_cnn_svhn.csv', sep=';', index=False)