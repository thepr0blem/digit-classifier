import numpy as np
from matplotlib import pyplot as plt
import pickle
import seaborn as sns
import random as rd

# Load the data
data_dir = r'./data/train_fixed.pkl'

file = open(data_dir, 'rb')
data = pickle.load(file)
file.close()

X, y = data

no_of_classes = X.shape[0]

# Labels dictionary

labels = {0: "6", 1: "P", 2: "O", 3: "V", 4: "W", 5: "3", 6: "A",
          7: "8", 8: "T", 9: "I", 10: "0", 11: "9", 12: "H", 13: "R",
          14: "N", 15: "7", 16: "K", 17: "L", 18: "G", 19: "4", 20: "Y",
          21: "C", 22: "E", 23: "J", 24: "5", 25: "1", 26: "S", 27: "2",
          28: "F", 29: "Z", 30: "U", 31: "Q", 32: "M", 33: "B", 34: "D"}


# Plotting number of classes

y_vec = y.reshape(y.shape[0], )

y_classes = sorted([labels[i] for i in y_vec])

sns.set(style="darkgrid")
count_plot = sns.countplot(y_classes, palette="Blues_d")

# plt.show()

# Comment - In the original data set there were 36 classes where one of them (class 30) had only one example
#           and was overlapped with class 14 (both were letter "N"). Single example from class 30 was moved to
#           class 14 and class 35 was renamed to 30.


def plot_samples(X, y):
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            n = rd.randint(0, X.shape[1])
            axarr[i, j].imshow(X[n].reshape(56, 56), cmap='gray')
            axarr[i, j].axis('off')
            axarr[i, j].set_title(labels[y[n][0]])


plot_samples(X, y)


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
