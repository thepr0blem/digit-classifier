# CNN Implementation using Keras

The purpose of this project is to develop convolutional neural network for written characters classification. 

## Introduction 

#### Project structure 

folders
files

#### Dataset overview
Training dataset consist of 30,134 examples of written characters divided into 35 classes - 10 digits and 25 letters. 

There is no class for letter "X". 

Each example is a 56x56 black/white image. Pixels are values 0 or 1. 

Training set is provided in form of numpy arrays with 3 136 columns (with pixel values) and one additional vector with class label. 

#### Content of this document 
1. Data loading, exploring and preprocessing the data
2. CNN modelling
3. Model evaluation

## 1. Loading and exploring the data

### 1.1 Loading the data
```python
# Load the data
data_dir = r'./data/train_fixed.pkl'

file = open(data_dir, 'rb')
data = pickle.load(file)
file.close()

X, y = data
```
Label dictionary 
```python
labels = {0: "6", 1: "P", 2: "O", 3: "V", 4: "W", 5: "3", 6: "A",
          7: "8", 8: "T", 9: "I", 10: "0", 11: "9", 12: "H", 13: "R",
          14: "N", 15: "7", 16: "K", 17: "L", 18: "G", 19: "4", 20: "Y",
          21: "C", 22: "E", 23: "J", 24: "5", 25: "1", 26: "S", 27: "2",
          28: "F", 29: "Z", 30: "U", 31: "Q", 32: "M", 33: "B", 34: "D"}
```          
Plotting classes distribution
```python
y_vec = y.reshape(y.shape[0], )

y_classes = sorted([labels[i] for i in y_vec])

sns.set(style="darkgrid")
count_plot = sns.countplot(y_classes, palette="Blues_d")

plt.show()
```
![Classes](https://github.com/thepr0blem/task/blob/master/pics/data_viz.png) 

### 1.2 Exploring data - samples

```python
def plot_samples(X, y):
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            n = rd.randint(0, X.shape[1])
            axarr[i, j].imshow(X[n].reshape(56, 56), cmap='gray')
            axarr[i, j].axis('off')
            axarr[i, j].set_title(labels[y[n][0]])


plot_samples(X, y)
```
![Classes](https://github.com/thepr0blem/task/blob/master/pics/samples.png) 

### 1.3 Preprocessing the data
#### 1.3.1 Reshaping 

```python
n_cols = X.shape[1]
img_size = int(np.sqrt(n_cols))
no_of_classes = len(np.unique(y, return_counts=True)[0])

X_cnn = X.reshape(X.shape[0], img_size, img_size, 1)
```

#### 1.3.2 Split into training and testing set 
The data is split in two steps. 
1. Split on training and testing sets in 90:10 proportion. 
2. Training set is then split on training and validation set in 90:10 proportion - this happens when fitting the model. 
```python
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y, test_size=0.1, random_state=42)
```

#### 1.3.3 Label encoding 

```python
y_train_cat_cnn = to_categorical(y_train_cnn)
y_test_cat_cnn = to_categorical(y_test_cnn)
```
      
## 2. Defining CNN structure
- describe the architecture
- how many conv layers, filters, kernels, padding
- max pooling, dropout 
- flatten
- dense 
- activation functions, optimizer 
- output function 
