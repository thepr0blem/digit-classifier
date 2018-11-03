""" --- WRITTEN CHARACTERS CLASSIFIER WORKFLOW --- """

# Required imports

import pickle
import random as rd
import numpy as np

from keras.models import load_model


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


def test_and_score(model, X_test, y_test):
    """Evaluates model on a given data set.

    Args:
        model: Instance of CNN model
        X_test: Testing data set.
        y_test: Labels of testing data set.

    Returns:
        Returns score.
    """
    score = model.evaluate(X_test, y_test, batch_size=64)
    return score



