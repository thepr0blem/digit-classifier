from keras.models import load_model


def predict(input_data):
    """This functions classifies given input

    Args:
        input_data: (n x 3136) array, where n - # of examples

    Returns:
        output_data: (n x 1) array with class labels
    """
    model_in = load_model(r"./models/CNN_FF_3.h5")

    prediction = model_in.predict(input_data.reshape(input_data.shape[0], 56, 56, 1))

    output_data = prediction.argmax(axis=1).reshape(len(prediction), 1)

    return output_data
