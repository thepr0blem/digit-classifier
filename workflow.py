"""This workflow goes through all the steps as presented in README"""

# Imports

import numpy as np
import pickle
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.utils import to_categorical
import keras

from src import modelling as mod
from src import visualization as vis
import predict as pred

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

# labels_new = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
#               7: "7", 8: "8", 9: "9", 10: "A", 11: "B", 12: "C", 13: "D",
#               14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J", 20: "K",
#               21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R",
#               28: "S", 29: "T", 30: "U", 31: "V", 32: "W", 33: "Y", 34: "Z"}
#
# inv_new = {v: k for k, v in labels_new.items()}
#
# y = np.array([inv_new[labels[i]] for i in list(y.reshape(1, len(y))[0])]).reshape(y.shape[0], 1)

# 1.2 Exploring data - sample
# Plotting number of classes

y_vec = y.reshape(y.shape[0], )

y_classes = sorted([labels[i] for i in y_vec])

# plt.figure(1)
# sns.set(style="darkgrid")
# count_plot = sns.countplot(y_classes, palette="Blues_d")


# vis.plot_samples(X, y, labels)


# 1.3 Preprocessing the data
# 1.3.1 Reshaping

n_cols = X.shape[1]
img_size = int(np.sqrt(n_cols))
no_of_classes = len(np.unique(y, return_counts=True)[0])

X_cnn = X.reshape(X.shape[0], img_size, img_size, 1)

# 1.3.1 Split into training and testing set

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

X_test = X_test_cnn.reshape(X_test_cnn.shape[0], img_size ** 2)

y_train_cat_cnn = to_categorical(y_train_cnn)
y_test_cat_cnn = to_categorical(y_test_cnn)

# 2 Defining CNN architecture
# 2.1 Defining CNN architecture
# 2.2 Hyperparameter tuning

# parameters_dct = {"no_of_filters": [8, 16, 32, 48, 64],
#                   "kern_size": [3, 4, 5, 6, 7],
#                   "max_pool": [2, 3],
#                   "dropout_perc": [0.1, 0.2, 0.3, 0.4, 0.5],
#                   "dense_size": [64, 128, 192, 256, 512, 1024],
#                   "optimizers": ["adam", "adamax", "nadam", "RMSProp"]
#                   }
#
# mod.run_random_search(X_train_cnn, y_train_cnn, parameters_dct, 20)
#
val_accs_list = np.load(r"./models/random_search/val_accs_list.npy")

# Print parameters with the highes val_accuracy

print(np.load(r"./models/random_search/params/params_dict_{}.npy".format(val_accs_list.argmax())))

# mod.create_model(X_train_cnn, y_train_cnn, it="77", no_of_filters=32, kern_size=3,
#                  max_p_size=3, drop_perc_conv=0.3, drop_perc_dense=0.2,
#                  dens_size=156, val_split_perc=0.1, no_of_epochs=30,
#                  optimizer="adam", random_search=False)


# 3 Model evaluation
# 3.1 Load model and evaluate on test data set

# model = keras.models.load_model(r"./models/CNN_v_F.h5")

# score = model.evaluate(X_test_cnn, y_test_cat_cnn, batch_size=64)

# 3.2 Confusion matrix

y_pred = pred.predict(X_test_cnn)

conf_mat = confusion_matrix(y_test_cnn, y_pred)

labels_list = [labels[i] for i in range(35)]

plt.figure(3)
vis.plot_conf_mat(conf_mat, labels_list, normalize=False)

# 3.3 Classification report

class_rep = classification_report(y_test_cnn, y_pred, target_names=labels_list)

print(class_rep)

# 3.4 Plot sample wrong classifications
vis.display_errors(X_test_cnn, y_test_cnn, y_pred, labels)