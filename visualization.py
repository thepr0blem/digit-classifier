import numpy as np
from matplotlib import pyplot as plt
import pickle

# Loading the data
data_dir = r'./data/train_fixed.pkl'

file = open(data_dir, 'rb')
data = pickle.load(file)

X, y = data


def display_n(x, y, n):
    """Display n-th example from data set

    Args:
        x: (i x j) array with i examples (each of them j features)
        y: (i x 1) vector with labels
        n: number of example to display

    Returns:
        Displays n-th example and returns class label.
    """
    img_size = int(np.sqrt(x.shape[1]))
    plt.imshow(x[n].reshape(img_size, img_size), cmap='gray')

    return y[n][0]
