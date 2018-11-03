import pickle
import numpy as np

# Loading the data
data_original = r'./data/train.pkl'
data_fixed = r'./data/train_fixed.pkl'

file = open(data_original, 'rb')
data = pickle.load(file)

X, y = data

class_count = np.unique(y, return_counts=True)

count_dict = dict(zip(class_count[0], class_count[1]))

print(count_dict[1])