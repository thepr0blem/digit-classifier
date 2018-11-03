from keras.models import load_model
import numpy as np


def predict(input_data):

    model_in = load_model(r"./models/CNN_1.h5")

    prediction = model_in.predict(input_data)

    output_data = prediction.argmax(axis=1).reshape(len(prediction), 1)

    return output_data
