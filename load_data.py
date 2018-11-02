""" --- WRITTEN CHARACTERS CLASSIFIER WORKFLOW --- """

# Required imports

import pickle
import random as rd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import EarlyStopping

from visualization import display_n
from augmentation import data_by_class

# Loading the data
data_dir = r'./data/train_fixed.pkl'

file = open(data_dir, 'rb')
data = pickle.load(file)

X, y = data

n_cols = X.shape[1]
img_size = int(np.sqrt(n_cols))
no_of_classes = len(np.unique(y, return_counts=True)[0])

X_cnn = X.reshape(X.shape[0], img_size, img_size, 1)

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y, test_size=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

y_train_cat_cnn = to_categorical(y_train_cnn)
y_test_cat_cnn = to_categorical(y_test_cnn)

model_path_nn = r".\models\NN_2.h5"
model_path_cnn = r".\models\CNN_1.h5"


def load_saved_model(path=model_path_nn):
    """Loads model using keras.load_model() function.

    Args:
        path: Model folder directory.
        filename: Name of the model file (.h5 file type)

    Returns:
        Keras model instance.
    """
    return load_model(path)


def test_and_score(model, x_test1=X_test, y_test1=y_test):
    """Evaluates model on a given data set.

    Args:
        model: Instance of CNN model
        x_test1: Testing data set.
        y_test1: Labels of testing data set.

    Returns:
        Returns score.
    """
    score = model.evaluate(x_test1, y_test1, batch_size=64)
    return score


parameters_dct = {"no_of_filters": [8, 16, 32, 48, 64],
                  "kern_size": [3, 4, 5],
                  "max_pool": [2, 3],
                  "dropout_perc": [0.05, 0.1, 0.2, 0.3, 0.4],
                  "dense_size": [64, 128, 192, 256, 320],
                  "optimizers": ["adam", "adamax", "nadam", "RMSProp"]
                  }


def run_random_search(params, no_of_searches=1):
    """Perform random search on hyper parameters list, saves models and validation accuracies.

    Args:
        params: Dictionary with hyperparameters for CNN random search.
        no_of_searches: How many times random search is executed.

    Returns:
        List of accuracies performed on validation set for each iteration.
    """
    val_accs_list = []

    for i in range(no_of_searches):
        # Creating a tuple for each iteration of random search with selected parameters

        params_dict = {"iteration": i + 1,
                       "no_of_filters": rd.choice(params["no_of_filters"]),
                       "kern_size": rd.choice(params["kern_size"]),
                       "max_pool": rd.choice(params["max_pool"]),
                       "dropout_perc_conv": rd.choice(params["dropout_perc"]),
                       "dropout_perc_dens": rd.choice(params["dropout_perc"]),
                       "dense_size": rd.choice(params["dense_size"]),
                       "optimizer": rd.choice(params["optimizers"]),
                       }

        np.save(r".\random_search\params\params_dict_{}.npy".format(i), params_dict)

        hist_dict = create_model(it=i + 1,
                                 no_of_filters=params_dict["no_of_filters"],
                                 kern_size=params_dict["kern_size"],
                                 max_p_size=params_dict["max_pool"],
                                 drop_perc_conv=params_dict["dropout_perc_conv"],
                                 drop_perc_dense=params_dict["dropout_perc_dens"],
                                 dens_size=params_dict["dense_size"],
                                 optimizer=params_dict["optimizer"]
                                 )

        val_accs_list.append(hist_dict['val_acc'][-1])

    np.save(r".\random_search\val_accs_list.npy", val_accs_list)

    return val_accs_list
