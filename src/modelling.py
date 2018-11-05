import numpy as np
import random as rd

from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical


def create_model(X, y, it=1, no_of_filters=32, kern_size=3,
                 max_p_size=3, drop_perc_conv=0.3, drop_perc_dense=0.2,
                 dens_size=128, val_split_perc=0.1, no_of_epochs=30,
                 optimizer="adam", random_search=False, batch_size=64):
    """Creates an architecture, train and saves CNN model.

    Returns:
        Dictionary with training report history.
    """

    y_train_cat = to_categorical(y)

    model = Sequential()

    model.add(Conv2D(no_of_filters,
                     kernel_size=(kern_size, kern_size),
                     activation='relu',
                     input_shape=(56, 56, 1),
                     padding='same'))

    model.add(Conv2D(no_of_filters,
                     kernel_size=(kern_size, kern_size),
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling2D((max_p_size, max_p_size)))
    model.add(Dropout(drop_perc_conv))

    model.add(Conv2D(no_of_filters,
                     kernel_size=(kern_size, kern_size),
                     activation='relu',
                     padding='same'))
    model.add(Conv2D(no_of_filters,
                     kernel_size=(kern_size, kern_size),
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling2D((max_p_size, max_p_size)))
    model.add(Dropout(drop_perc_conv))

    model.add(Flatten())

    model.add(Dense(dens_size, activation='relu'))
    model.add(Dropout(drop_perc_dense))

    model.add(Dense(35, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping_monitor = EarlyStopping(patience=5)
    rlrop = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, verbose=1, min_lr=0.00001)

    history = model.fit(X,
                        y_train_cat,
                        validation_split=val_split_perc,
                        epochs=no_of_epochs,
                        callbacks=[early_stopping_monitor, rlrop],
                        batch_size=batch_size)

    history_dict = history.history

    if random_search:

        np.save(r"./models/random_search/hist/history_dict_{}.npy".format(it), history_dict)
        model.save(r"./models/random_search/models/CNN_{}.h5".format(it))

    else:

        np.save(r"./logs/history_dict_{}.npy".format(it), history_dict)
        model.save(r"./models/CNN_model_{}.h5".format(it))

    return history_dict


def run_random_search(X, y, params, no_of_searches=1):
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
                       "batch_size": rd.choice(params["batch_size"])
                       }

        np.save(r"./models/random_search/params/params_dict_{}.npy".format(i), params_dict)

        hist_dict = create_model(X, y,
                                 it=i + 1,
                                 no_of_filters=params_dict["no_of_filters"],
                                 kern_size=params_dict["kern_size"],
                                 max_p_size=params_dict["max_pool"],
                                 drop_perc_conv=params_dict["dropout_perc_conv"],
                                 drop_perc_dense=params_dict["dropout_perc_dens"],
                                 dens_size=params_dict["dense_size"],
                                 optimizer=params_dict["optimizer"],
                                 random_search=True,
                                 batch_size=params_dict["batch_size"],
                                 no_of_epochs=3,
                                 val_split_perc=0.1
                                 )

        val_accs_list.append(hist_dict['val_acc'][-1])

    np.save(r"./models/random_search/val_accs_list.npy", val_accs_list)

    return val_accs_list


