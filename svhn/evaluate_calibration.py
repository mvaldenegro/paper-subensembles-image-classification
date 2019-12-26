import numpy as np
import h5py
import pandas as pd

from svhn_io import load_svhn
from keras_uncertainty.utils import classifier_calibration_curve, classifier_calibration_error

EPSILON = 1e-10

def load_hdf5_data(filename):
    inp = h5py.File(filename, "r")
    preds = inp["preds"][...]

    inp.close()

    return preds

NUM_ENSEMBLES = 15
NUM_BINS=7

#IOD_FILE_PATTERN = "cnn_svhn-num_ens-{}-preds.hdf5"
#OUTPUT_PATTERN = "svhn-calibration-sub-deepensembles_1_num-ens-{}_cnn_svhn.csv"

IOD_FILE_PATTERN = "deepensembles-cnn_svhn-num_ens-{}-preds.hdf5"
OUTPUT_PATTERN = "svhn-calibration-deepensembles-num-ens-{}_cnn_svhn.csv"

if __name__ == "__main__":
    for num_ens in range(1, NUM_ENSEMBLES + 1):
        (_, __), (___, y_true) = load_svhn()
        y_true = y_true.flatten()

        y_probs = load_hdf5_data(IOD_FILE_PATTERN.format(num_ens))
        y_confs = np.max(y_probs, axis=1)
        y_pred = np.argmax(y_probs, axis=1)

        curve_conf, curve_acc = classifier_calibration_curve(y_pred, y_true, y_confs, num_bins=NUM_BINS)
        error = classifier_calibration_error(y_pred, y_true, y_confs, num_bins=NUM_BINS)

        print("Processing calibration curve for {} ensembles. Error: {}".format(num_ens, error))

        output_df = pd.DataFrame(data={"conf": curve_conf, "acc": curve_acc})
        output_df.to_csv(OUTPUT_PATTERN.format(num_ens), sep=';', index=False)