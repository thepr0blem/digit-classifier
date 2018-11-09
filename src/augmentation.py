from keras.preprocessing.image import ImageDataGenerator
import pickle
import numpy as np


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
    datagen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 rotation_range=10)

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

    classes = np.unique(y)
    augmented_req_x = np.zeros((1, 56 * 56)).astype(int)
    augmented_req_y = np.zeros((1, 1)).astype(int)

    for i in classes:

        X_lab, y_lab = data_by_class(data, i)
        print("Data augmenting for class {}".format(i))
        X_aug, y_aug = data_aug(X_lab, y_lab, 300)
        print("Finished, loading next class. ")
        augmented_req_x = np.vstack((augmented_req_x, X_aug))
        augmented_req_y = np.vstack((augmented_req_y, y_aug))

    print("Finished.")

    return np.vstack((X, augmented_req_x)), np.vstack((y, augmented_req_y))


if __name__ == "__main__":

    data_dir = r'./data/train_fix.pkl'
    data_dir_aug = r'./data/train_fix_aug.pkl'

    with open(data_dir, 'rb') as f:
        data = pickle.load(f)

    X, y = data

    data_new = gen_missing_data(data)

    output = open(data_dir_aug, 'wb')
    pickle.dump(data_new, output)
    output.close()
