from keras.preprocessing.image import ImageDataGenerator
from keras import backend as ker
import pickle
from matplotlib import pyplot as plt
import numpy as np

data_dir = r'./data/train_fixed.pkl'
data_dir_aug = r'./data/train_aug_filled_fixed.pkl'

file = open(data_dir, 'rb')
data = pickle.load(file)

X, y = data

# max_count = np.max(np.unique(y, return_counts=True))
# no_of_classes = len(np.unique(y, return_counts=True)[0])
# n_cols = X.shape[1]
# img_size = int(np.sqrt(n_cols))
# ker.set_image_dim_ordering('th')
# reshape to be [samples][pixels][width][height]

# X_train = X.reshape(X.shape[0], 1, img_size, img_size)


def data_by_class(data, label):
    """Selects part of data set by specified label

    Args:
        data: Tuple with two numpy arrays for data set and appropriate labels
        label: Data for which label the function selects

    Returns:
        Tuple with two vectors (X, y) containing only data for specified label
    """

    X, y = data
    y_trans = y.reshape(1, y.shape[0])
    indices = np.where(y_trans[0] == label)

    return X[indices, :].reshape(len(indices[0]), X.shape[1]), y[indices, :].reshape(len(indices[0]), 1)


def data_aug(X_to_aug, y_to_aug, count_add=1000):
    """Generate augmented images.

    Args:
        X_to_aug: Input array with images
        y_to_aug: Vector with labels for input images array

    Returns:
        Two arrays(X, y) containing count_add number of generated augmented images.
    """

    img_size = int(np.sqrt(X_to_aug.shape[1]))

    X_resh = X_to_aug.reshape(X_to_aug.shape[0], img_size, img_size, 1)

    # Initialize ImageDataGen
    datagen = ImageDataGenerator(width_shift_range=0.05,
                                 height_shift_range=0.05,
                                 rotation_range=15)

    augmented_x = []
    augmented_y = []

    for i in range(count_add):      # draw required number of augmented data points

        X_aug_sq, y_aug = next(datagen.flow(X_resh, y_to_aug, batch_size=1, shuffle=True))
        X_aug = np.round(X_aug_sq.reshape(1, img_size ** 2), 0).astype(int)

        augmented_x.append(X_aug)
        augmented_y.append(y_aug)

    return np.array(augmented_x).reshape(count_add, img_size **2), np.array(augmented_y).reshape(count_add, 1)


def gen_missing_data(data):

    X, y = data

    max_count = np.max(np.unique(y, return_counts=True))
    req_count_vec = (np.unique(y, return_counts=True)[1] - max_count) * (-1) + 1
    no_of_classes = len(np.unique(y, return_counts=True)[0])

    augmented_req_x = np.zeros((1, 56 * 56)).astype(int)
    augmented_req_y = np.zeros((1, 1)).astype(int)

    for i in range(no_of_classes):

        X_lab, y_lab = data_by_class(data, i)
        print("Data augmenting for class {}".format(i))
        X_aug, y_aug = data_aug(X_lab, y_lab, req_count_vec[i])
        print("Finished, loading next class. ")
        augmented_req_x = np.vstack((augmented_req_x, X_aug))
        augmented_req_y = np.vstack((augmented_req_y, y_aug))

    print("Finished.")

    return np.vstack((X, augmented_req_x)), np.vstack((y, augmented_req_y))


# data_new = gen_missing_data(data)
#
# output = open(data_dir_aug, 'wb')
# pickle.dump(data_new, output)
# output.close()


def gen_disp(data_new):
    for i in range(36):
        for k in range(1, 1101, 100):
            print(i)
            yield display_n(data_by_class(data_new, i)[0], data_by_class(data_new, i)[1], k)