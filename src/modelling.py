import numpy as np

from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


def create_model(X_train, y_train, it=1, no_of_filters=32, kern_size=5,
                 max_p_size=2, drop_perc_conv=0.2, drop_perc_dense=0.4,
                 dens_size=256, val_split_perc=0.2, no_of_epochs=5,
                 optimizer="adam", random_search=False):

    """Creates an architecture, train and saves CNN model.

    Returns:
        Dictionary with training report history.
    """

    img_size = int(np.sqrt(X_train.shape[1]))
    num_classes = len(np.unique(y_train, return_counts=True)[0])
    y_train_cat = to_categorical(y_train)

    model = Sequential()

    model.add(Conv2D(no_of_filters,
                     kernel_size=(kern_size, kern_size),
                     activation='relu',
                     input_shape=(img_size, img_size, 1),
                     padding='same'))

    model.add(Conv2D(no_of_filters,
                     kernel_size=(kern_size, kern_size),
                     activation='relu',
                     padding='valid'))
    model.add(MaxPooling2D(pool_size=(max_p_size, max_p_size)))
    model.add(Dropout(drop_perc_conv))

    model.add(Conv2D(2 * no_of_filters,
                     kernel_size=(kern_size-2, kern_size-2),
                     activation='relu',
                     input_shape=(img_size, img_size, 1),
                     padding='same'))
    model.add(Conv2D(2 * no_of_filters,
                     kernel_size=(kern_size-2, kern_size-2),
                     activation='relu',
                     input_shape=(img_size, img_size, 1),
                     padding='valid'))
    model.add(MaxPooling2D(pool_size=(max_p_size, max_p_size)))
    model.add(Dropout(drop_perc_conv))

    model.add(Flatten())

    model.add(Dense(dens_size, activation='relu'))
    model.add(Dropout(drop_perc_dense))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping_monitor = EarlyStopping(patience=5)

    history = model.fit(X_train,
                        y_train_cat,
                        validation_split=val_split_perc,
                        epochs=no_of_epochs,
                        callbacks=[early_stopping_monitor],
                        batch_size=128
                        )

    history_dict = history.history

    if random_search:

        np.save(r".\random_search\hist\history_dict_{}.npy".format(it), history_dict)
        model.save(r".\random_search\models\CNN_{}.h5".format(it))

    else:

        np.save(r".\logs\history_dict_{}.npy".format(it), history_dict)
        model.save(r".\logs\CNN_model_{}.h5".format(it))

    return history_dict


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