import numpy as np
import h5py
import pandas as pd

from skimage. io import imsave
from keras.datasets import cifar10
from svhn_io import load_svhn

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EPSILON = 1e-10

def compute_entropy_scores(data, axis=-1):
    return np.sum(-data * np.log(data + EPSILON), axis=axis)

def load_hdf5_data(filename):
    inp = h5py.File(filename, "r")
    preds = inp["preds"][...]

    inp.close()

    return preds

def plot_probabilities(output_filename, class_names, probs, correct_class=None):
    assert len(class_names) == len(probs)

    fig, ax = plt.subplots()
    x = list(range(len(class_names)))

    

    plot = plt.barh(x, probs, height=1.0, edgecolor='darkblue')

    anchor = max(probs)
    for idx,rect in enumerate(plot):            
        ax.text(anchor, rect.get_y() + rect.get_height() / 2.0,
                class_names[idx],
                ha='right', va='bottom', backgroundcolor='lightgrey', color='black')

    ax.set_frame_on(False)
    ax.get_yaxis().set_visible(False)

    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))

    #if correct_class is not None:
    #    plt.title(correct_class)

    plt.savefig(output_filename, format="pdf", bbox_inches='tight')
    #plt.show()


SVHN_CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
CIFAR10_CLASS_NAMES = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

ENSEMBLES = [1, 5, 10, 15]
NUM_OUTPUTS = 5

SE_IOD_FILE_PATTERN = "cnn_svhn-num_ens-{}-preds.hdf5"
SE_OOD_FILE_PATTERN = "cnn_svhn-num_ens-{}-ood-preds.hdf5"

OUTPUT_PATTERN = "ood-roc-sub-deepensembles_1_num-ens-{}_cnn_svhn.csv"

DE_IOD_FILE_PATTERN = "deepensembles-cnn_svhn-num_ens-{}-preds.hdf5"
DE_OOD_FILE_PATTERN = "deepensembles-cnn_svhn-num_ens-{}-ood-preds.hdf5"

#OUTPUT_PATTERN = "ood-roc-deepensembles-num-ens-{}_cnn_svhn.csv"

DOMINANT_ENS = 15

if __name__ == "__main__":
    (_, _), (x_cifar10, y_cifar10) = cifar10.load_data()
    (_, _), (x_svhn, y_svhn) = load_svhn()

    se_raw_iod_dom = load_hdf5_data(SE_IOD_FILE_PATTERN.format(DOMINANT_ENS))
    se_raw_ood_dom = load_hdf5_data(SE_OOD_FILE_PATTERN.format(DOMINANT_ENS))

    se_entropy_iod_dom = compute_entropy_scores(se_raw_iod_dom)
    se_entropy_ood_dom = compute_entropy_scores(se_raw_ood_dom)

    iod_top_dom = se_entropy_iod_dom.argsort()[::-1][:NUM_OUTPUTS]
    ood_top_dom = se_entropy_ood_dom.argsort()[::-1][:NUM_OUTPUTS]

    for num_ens in ENSEMBLES:
        se_raw_iod = load_hdf5_data(SE_IOD_FILE_PATTERN.format(num_ens))
        se_raw_ood = load_hdf5_data(SE_OOD_FILE_PATTERN.format(num_ens))

        de_raw_iod = load_hdf5_data(DE_IOD_FILE_PATTERN.format(num_ens))
        de_raw_ood = load_hdf5_data(DE_OOD_FILE_PATTERN.format(num_ens))

        #print("Raw IOD predictions shape: {}".format(raw_iod.shape))
        #print("Raw OOD predictions shape: {}".format(raw_ood.shape))

        se_entropy_iod = compute_entropy_scores(se_raw_iod)
        se_entropy_ood = compute_entropy_scores(se_raw_ood)

        de_entropy_iod = compute_entropy_scores(de_raw_iod)
        de_entropy_ood = compute_entropy_scores(de_raw_ood)
                
        print("{} Ensembles".format(num_ens))
        print("Mean IOD Entropy: {:.4f} Mean OOD Entropy: {:.4f}".format(np.mean(se_entropy_iod), np.mean(se_entropy_ood)))

        #iod_top = se_entropy_iod.argsort()[::-1][:NUM_OUTPUTS]
        #ood_top = se_entropy_ood.argsort()[::-1][:NUM_OUTPUTS]

        #iod_bottom = se_entropy_iod.argsort()[::-1][-NUM_OUTPUTS:]
        #ood_bottom = se_entropy_ood.argsort()[::-1][-NUM_OUTPUTS:]

        for idx in iod_top_dom:
            preds = se_raw_iod[idx, :]
            preds = ",".join(["{:.2f}".format(p) for p in preds])
            print("SVHN Image {} has entropy {:.2f} and predictions {}".format(idx, se_entropy_iod[idx], preds))

            image = x_svhn[idx, :, :, :]
            imsave("iod-svhn-sub-ensembles-{}-top-idx{}-entropy{:.2f}.png".format(num_ens, idx, se_entropy_iod[idx]), image)
            probs = se_raw_iod[idx, :]
            correct_class_idx = SVHN_CLASS_NAMES[y_svhn[idx][0]]
            correct_class = "{} - Class {}".format("Sub-Ensembles", correct_class_idx)

            plot_probabilities("iod-svhn-sub-ensembles-{}-top-idx{}-probabilities.pdf".format(num_ens, idx), SVHN_CLASS_NAMES, probs, correct_class)

            de_probs = de_raw_iod[idx, :]
            correct_class = "{} - Class {}".format("Deep Ensembles", correct_class_idx)

            plot_probabilities("iod-svhn-deep-ensembles-{}-top-idx{}-probabilities.pdf".format(num_ens, idx), SVHN_CLASS_NAMES, de_probs, correct_class)

        for idx in ood_top_dom:
            print("CIFAR10 Image {} has entropy {:.2f}".format(idx, se_entropy_ood[idx]))

            image = x_cifar10[idx, :, :, :]
            imsave("ood-cifar10-sub-ensembles-{}-top-idx{}-entropy{:.2f}.png".format(num_ens, idx, se_entropy_ood[idx]), image)
            probs = se_raw_ood[idx, :]
            correct_class_idx = SVHN_CLASS_NAMES[y_svhn[idx][0]]
            correct_class = "{} - Class {}".format("Sub-Ensembles", correct_class_idx)

            plot_probabilities("ood-svhn-sub-ensembles-{}-top-idx{}-probabilities.pdf".format(num_ens, idx), SVHN_CLASS_NAMES, probs, correct_class)

            de_probs = de_raw_ood[idx, :]
            correct_class = "{} - Class {}".format("Deep Ensembles", correct_class_idx)

            plot_probabilities("ood-svhn-deep-ensembles-{}-top-idx{}-probabilities.pdf".format(num_ens, idx), SVHN_CLASS_NAMES, de_probs, correct_class)

        #for idx in iod_bottom:
        #    print("SVHN Image {} has entropy {:.2f}".format(idx, entropy_iod[idx]))
        #    image = x_svhn[idx, :, :, :]
        #    imsave("iod-svhn-sub-ensembles-{}-bottom-idx{}-entropy{:.2f}.png".format(num_ens, idx, entropy_iod[idx]), image)

        #for idx in ood_bottom:
        #    print("CIFAR10 Image {} has entropy {:.2f}".format(idx, entropy_ood[idx]))

        #    image = x_cifar10[idx, :, :, :]
        #    imsave("ood-cifar10-sub-ensembles-{}-bottom-1idx{}-entropy{:.2f}.png".format(num_ens, idx, entropy_ood[idx]), image)

            
