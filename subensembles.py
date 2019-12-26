import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from keras_uncertainty.utils import numpy_negative_log_likelihood

def evaluate_subensemble_layers(num_ensembles, x_train, y_train, x_test, y_test, train_model_fn, train_submodel_fn, max_layers, subensemble_layers):
    results = pd.DataFrame(columns = ['num_ensembles', 'error', 'nll'])

    ensemble_models = []
    base_model = None

    for num_ens in num_ensembles:

        # First member of the ensemble, make base model
        if num_ens == 1:
            model, base_model = train_model_fn(x_train, y_train, max_layers - subensemble_layers, subensemble_layers)
        else:
            model = train_submodel_fn(base_model, x_train, y_train, max_layers - subensemble_layers, subensemble_layers)

        #Add one member to the ensemble    
        ensemble_models.append(model)
            
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

    return results