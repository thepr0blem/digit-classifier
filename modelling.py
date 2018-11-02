import numpy as np

from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.utils import to_categorical


def create_model(X_train, y_train, it=1, no_of_filters=32, kern_size=5, max_p_size=2, drop_perc_conv=0.2, drop_perc_dense=0.4,
                 dens_size=256, val_split_perc=0.2, no_of_epochs=5, optimizer="adam"):

    """Creates an architecture, train and saves CNN model.

    Returns:
        Dictionary with training report history.
    """

    img_size = int(np.sqrt(X_train.shape[1]))

    model = Sequential()

    model.add(Conv2D(no_of_filters,
                     kernel_size=(kern_size, kern_size),
                     activation='relu',
                     input_shape=(img_size, img_size, 1),
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(max_p_size, max_p_size)))
    model.add(Dropout(drop_perc_conv))

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

    model.add(Dense(int(dens_size/2), activation='relu'))
    model.add(Dropout(drop_perc_dense))

    model.add(Dense(y_train_cat.shape[1], activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping_monitor = EarlyStopping(patience=5)

    history = model.fit(X_train_cnn,
                        y_train_cat_cnn,
                        validation_split=val_split_perc,
                        epochs=no_of_epochs,
                        callbacks=[early_stopping_monitor],
                        batch_size=128
                        )

    history_dict = history.history

    # for ordinary use
    np.save(r".\logs\history_dict_{}.npy".format(it), history_dict)
    model.save(model_path_cnn)

    # for random search
    # np.save(r".\random_search\hist\history_dict_{}.npy".format(it), history_dict)
    # model.save(r".\random_search\models\CNN_{}.h5".format(it))

    return history_dict