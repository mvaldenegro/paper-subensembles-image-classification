import numpy as np
import h5py
import pandas as pd

from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import gaussian_kde

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

ENSEMBLES = [1, 5, 10, 15]

#IOD_FILE_PATTERN = "cnn_svhn-num_ens-{}-preds.hdf5"
#OOD_FILE_PATTERN = "cnn_svhn-num_ens-{}-ood-preds.hdf5"

#OUTPUT_PATTERN = "entropy-distribution-sub-deepensembles_1_num-ens-{}_cnn_svhn.csv"

IOD_FILE_PATTERN = "deepensembles-cnn_svhn-num_ens-{}-preds.hdf5"
OOD_FILE_PATTERN = "deepensembles-cnn_svhn-num_ens-{}-ood-preds.hdf5"

OUTPUT_PATTERN = "entropy-distribution-deepensembles-num-ens-{}_cnn_svhn.csv"

HISTOGRAM_BINS = 15

if __name__ == "__main__":
    for num_ens in ENSEMBLES:
        raw_iod = load_hdf5_data(IOD_FILE_PATTERN.format(num_ens))
        raw_ood = load_hdf5_data(OOD_FILE_PATTERN.format(num_ens))

        entropy_iod = compute_entropy_scores(raw_iod)
        entropy_ood = compute_entropy_scores(raw_ood)

        print("{} Ensembles, ID Minimum Entropy: {:.2f} Maximum Entropy: {:.2f} Mean Entropy {:.2f}".format(num_ens, min(entropy_iod), max(entropy_iod), np.mean(entropy_iod)))
        print("{} Ensembles, OOD Minimum Entropy: {:.2f} Maximum Entropy: {:.2f} Mean Entropy {:.2f}".format(num_ens, min(entropy_ood), max(entropy_ood), np.mean(entropy_ood)))

        plot_min = min(min(entropy_iod), min(entropy_ood)) - 0.05
        plot_max = max(max(entropy_iod), max(entropy_ood)) + 0.05

        domain = np.linspace(plot_min, plot_max, num=30)

        id_kde = gaussian_kde(entropy_iod)
        ood_kde = gaussian_kde(entropy_ood)

        id_density = id_kde.evaluate(domain)
        ood_density = ood_kde.evaluate(domain)

        output_df = pd.DataFrame(data={"id_entropy": domain, "id_density": id_density,
                                       "ood_entropy": domain, "ood_density": ood_density})

        output_df.to_csv(OUTPUT_PATTERN.format(num_ens), sep=';', index=False)