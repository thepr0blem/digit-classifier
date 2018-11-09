"""This workflow goes through all the steps as presented in README"""

# Imports

import numpy as np
import pickle
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import keras

from src import modelling as mod
from src import visualization as vis
import predict as pred

# 2 Loading and exploring the data
# 2.1 Load the data

data_dir = r'./data/train_fix.pkl'

with open(data_dir, 'rb') as f:
    data = pickle.load(f)

X, y = data

# Labels dictionary

labels = {0: "6", 1: "P", 2: "O", 3: "V", 4: "W", 5: "3", 6: "A",
          7: "8", 8: "T", 9: "I", 10: "0", 11: "9", 12: "H", 13: "R",
          14: "N", 15: "7", 16: "K", 17: "L", 18: "G", 19: "4", 20: "Y",
          21: "C", 22: "E", 23: "J", 24: "5", 25: "1", 26: "S", 27: "2",
          28: "F", 29: "Z", 30: "?", 31: "Q", 32: "M", 33: "B", 34: "D", 35: "U"}

# 2.2 Exploring data - samples
# Plotting number of classes

y_classes = sorted([labels[i] for i in y.reshape(y.shape[0], )])

plt.figure(1)
sns.set(style="darkgrid")
count_plot = sns.countplot(y_classes, palette="Blues_d")

vis.plot_samples(X, y, labels)

# 2.3 Preprocessing the data
# 2.3.1 Reshaping

n_cols = X.shape[1]
img_size = int(np.sqrt(n_cols))
no_of_classes = len(np.unique(y, return_counts=True)[0])

X_cnn = X.reshape(X.shape[0], img_size, img_size, 1)

# 2.3.2 Split into training and testing set

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

X_test = X_test_cnn.reshape(X_test_cnn.shape[0], img_size ** 2)

# 2.3.3 Label encoding
y_train_cat_cnn = to_categorical(y_train_cnn)
y_test_cat_cnn = to_categorical(y_test_cnn)

# 3 Defining CNN architecture
# 3.1 Defining CNN architecture
# 3.2 Hyperparameter tuning

parameters_dct = {"no_of_filters": [8, 16, 24, 32, 40, 48, 56, 64],
                  "kern_size": [3, 4, 5, 6, 7],
                  "max_pool": [2, 3, 4],
                  "dropout_perc": [0.1, 0.2, 0.3, 0.4, 0.5],
                  "dense_size": [64, 128, 192, 256, 384, 512],
                  "optimizers": ["adam", "adamax", "nadam", "RMSProp"],
                  "batch_size": [16, 32, 32, 48, 64, 96, 128]
                  }

# mod.run_random_search(X_train_cnn, y_train_cnn, parameters_dct, 70)

val_accs_list = np.load(r"./models/random_search/val_accs_list.npy")

# Print parameters with the highes val_accuracy

top_three_indices = np.argsort(val_accs_list)[::-1][:3]

for i in top_three_indices:

    print(np.load(r"./models/random_search/params/params_dict_{}.npy".format(i)))


# mod.create_model(X_train_cnn, y_train_cnn, it="1", no_of_filters=24, kern_size=3,
#                  max_p_size=3, drop_perc_conv=0.1, drop_perc_dense=0.4,
#                  dens_size=384, val_split_perc=0.1, no_of_epochs=40,
#                  optimizer="adam", random_search=False, batch_size=24)
#
# mod.create_model(X_train_cnn, y_train_cnn, it="2", no_of_filters=56, kern_size=6,
#                  max_p_size=3, drop_perc_conv=0.3, drop_perc_dense=0.4,
#                  dens_size=384, val_split_perc=0.1, no_of_epochs=40,
#                  optimizer="adam", random_search=False, batch_size=24)
#
# mod.create_model(X_train_cnn, y_train_cnn, it="3", no_of_filters=40, kern_size=5,
#                  max_p_size=3, drop_perc_conv=0.3, drop_perc_dense=0.3,
#                  dens_size=512, val_split_perc=0.1, no_of_epochs=40,
#                  optimizer="adam", random_search=False, batch_size=24)

# 4 Model evaluation
# 4.1 Load model and evaluate on test data set

# model_1 = keras.models.load_model(r"./models/CNN_FF_1.h5")
# model_2 = keras.models.load_model(r"./models/CNN_FF_2.h5")
model_3 = keras.models.load_model(r"./models/CNN_FF_3.h5")

# score_1 = model_1.evaluate(X_test_cnn, y_test_cat_cnn, batch_size=64)
# score_2 = model_2.evaluate(X_test_cnn, y_test_cat_cnn, batch_size=64)
# score_3 = model_3.evaluate(X_test_cnn, y_test_cat_cnn, batch_size=64)

# print("Model_1: val_acc - {}".format(np.round(score_1[1] * 100, 2)),
#       "\nModel_2: val_acc - {}".format(np.round(score_2[1] * 100, 2)),
#       "\nModel_3: val_acc - {}".format(np.round(score_3[1] * 100, 2)))

# 4.2 Confusion matrix

y_pred = pred.predict(X_test_cnn)

conf_mat = confusion_matrix(y_test_cnn, y_pred, labels=list(labels.keys()))

labels_list = [labels[i] for i in range(36)]

plt.figure(3)
vis.plot_conf_mat(conf_mat, labels_list, normalize=False)

# 4.3 Plot sample wrong classifications

vis.display_errors(X_test_cnn, y_test_cnn, y_pred, labels)