import numpy as np
import h5py
import pandas as pd

from sklearn.metrics import roc_curve, roc_auc_score

EPSILON = 1e-10

def compute_entropy_scores(data, axis=-1):
    return np.sum(-data * np.log(data + EPSILON), axis=axis)

def compute_roc(data_iod, data_ood):
    scores = np.concatenate([data_iod, data_ood], axis=0)
    labels = np.concatenate([np.zeros_like(data_iod), np.ones_like(data_ood)], axis=0)

    norm_scores = scores - min(scores) / (max(scores) - min(scores))

    #print("Data {} Labels {}".format(data.shape, labels.shape))

    auc = roc_auc_score(labels, scores)    
    fpr, tpr, threshs = roc_curve(labels, norm_scores, drop_intermediate=True)

    return auc, fpr, tpr, threshs

def load_hdf5_data(filename):
    inp = h5py.File(filename, "r")
    preds = inp["preds"][...]

    inp.close()

    return preds

NUM_ENSEMBLES = 15

#IOD_FILE_PATTERN = "cnn_svhn-num_ens-{}-preds.hdf5"
#OOD_FILE_PATTERN = "cnn_svhn-num_ens-{}-ood-preds.hdf5"

#OUTPUT_PATTERN = "ood-roc-sub-deepensembles_1_num-ens-{}_cnn_svhn.csv"

IOD_FILE_PATTERN = "deepensembles-cnn_svhn-num_ens-{}-preds.hdf5"
OOD_FILE_PATTERN = "deepensembles-cnn_svhn-num_ens-{}-ood-preds.hdf5"

OUTPUT_PATTERN = "ood-roc-deepensembles-num-ens-{}_cnn_svhn.csv"

if __name__ == "__main__":
    for num_ens in range(1, NUM_ENSEMBLES + 1):
        raw_iod = load_hdf5_data(IOD_FILE_PATTERN.format(num_ens))
        raw_ood = load_hdf5_data(OOD_FILE_PATTERN.format(num_ens))

        entropy_iod = compute_entropy_scores(raw_iod)
        entropy_ood = compute_entropy_scores(raw_ood)

        auc, fpr, tpr, threshs = compute_roc(entropy_iod, entropy_ood)
                
        print("{} Ensembles. IOD/OOD AUC: {:.4f}".format(num_ens, auc))
        print("Mean IOD Entropy: {:.4f} Mean OOD Entropy: {:.4f}".format(np.mean(entropy_iod), np.mean(entropy_ood)))

        output_df = pd.DataFrame(data={"fpr": fpr, "tpr": tpr, "thresh": threshs})
        output_df.to_csv(OUTPUT_PATTERN.format(num_ens), sep=';', index=False)