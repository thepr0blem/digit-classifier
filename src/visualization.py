import numpy as np
from matplotlib import pyplot as plt
import random as rd
import seaborn as sns


def plot_samples(X, y, labels):
    """Display 3x3 plot with sample images from X, y dataset.

    Args:
        X: (i x j) array with i examples (each of them j features)
        y: (i x 1) vector with labels
        n: dict with {class: label} structure

    Returns:
        Displays n-th example and returns class label.
    """
    f, plarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            n = rd.randint(0, X.shape[1])
            plarr[i, j].imshow(X[n].reshape(56, 56), cmap='gray')
            plarr[i, j].axis('off')
            plarr[i, j].set_title(labels[y[n][0]])


def display_n(X, y, n):
    """Display n-th example from data set

    Args:
        X: (i x j) array with i examples (each of them j features)
        y: (i x 1) vector with labels
        n: number of example to display

    Returns:
        Displays n-th example and returns class label.
    """
    img_size = int(np.sqrt(X.shape[1]))
    plt.imshow(X[n].reshape(img_size, img_size), cmap='gray')

    return y[n][0]


def plot_conf_mat(conf_mat, label_list, normalize=False):

    if normalize:

        conf_mat = conf_mat.astype(float) / conf_mat.sum(axis=1)[:, np.newaxis]

    fmt = '.2f' if normalize else 'd'
    sns.heatmap(conf_mat, annot=True, fmt=fmt,
                cmap="Blues", cbar=False, xticklabels=label_list,
                yticklabels=label_list, robust=True)
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
