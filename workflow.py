"""This workflow goes through all the steps as presented in README"""

# Imports

import numpy as np
import pickle
import seaborn as sns
import random as rd

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.utils import to_categorical

from src import visualization as vis
from src import modelling as mdl

# 1 Loading and exploring the data
# 1.1 Load the data

data_dir = r'./data/train_fixed.pkl'

file = open(data_dir, 'rb')
data = pickle.load(file)
file.close()

X, y = data

# Labels dictionary

labels = {0: "6", 1: "P", 2: "O", 3: "V", 4: "W", 5: "3", 6: "A",
          7: "8", 8: "T", 9: "I", 10: "0", 11: "9", 12: "H", 13: "R",
          14: "N", 15: "7", 16: "K", 17: "L", 18: "G", 19: "4", 20: "Y",
          21: "C", 22: "E", 23: "J", 24: "5", 25: "1", 26: "S", 27: "2",
          28: "F", 29: "Z", 30: "U", 31: "Q", 32: "M", 33: "B", 34: "D"}


# 1.2 Exploring data - sample
# Plotting number of classes

y_vec = y.reshape(y.shape[0], )

y_classes = sorted([labels[i] for i in y_vec])

sns.set(style="darkgrid")
count_plot = sns.countplot(y_classes, palette="Blues_d")

# plt.show()

# Comment - In the original data set there were 36 classes where one of them (class 30) had only one example
#           and was overlapped with class 14 (both were letter "N"). Single example from class 30 was moved to
#           class 14 and class 35 was renamed to 30.

vis.plot_samples(X, y, labels)

# 1.3 Preprocessing the data
# 1.3.1 Reshaping

n_cols = X.shape[1]
img_size = int(np.sqrt(n_cols))
no_of_classes = len(np.unique(y, return_counts=True)[0])

X_cnn = X.reshape(X.shape[0], img_size, img_size, 1)

# 1.3.1 Split into training and testing set

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

y_train_cat_cnn = to_categorical(y_train_cnn)
y_test_cat_cnn = to_categorical(y_test_cnn)

# 2 Defining CNN architecture

mdl.create_model(X_train_cnn, y_train_cnn)