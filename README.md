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
### 1.2 Exploring data - samples
Plotting classes distribution
```python
y_vec = y.reshape(y.shape[0], )

y_classes = sorted([labels[i] for i in y_vec])

sns.set(style="darkgrid")
count_plot = sns.countplot(y_classes, palette="Blues_d")

plt.show()
```
![Classes](https://github.com/thepr0blem/task/blob/master/pics/data_viz.png) 

```python
def plot_samples(X, y):
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            n = rd.randint(0, X.shape[1])
            axarr[i, j].imshow(X[n].reshape(56, 56), cmap='gray')
            axarr[i, j].axis('off')
            axarr[i, j].set_title(labels[y[n][0]])


vis.plot_samples(X, y)
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
      
## 2. Building CNN 
### 2.1 Defining CNN architecture

To implement convolutional neural network I used **Keras** API (which is user friendly framework built on top of Tensorflow). I used Sequential model which is ordered hierarchy of layers. Layers are ordered as follows: 
  - **Conv2D** - conv. layer 
    - filters - 32
    - kernel_size - 3 x 3
    - activation - 'relu' 
    - input shape - 4D tensor - (n, 56, 56, 1), where (number of examples, img_size, img_size, no_of_channels) 
  - **Conv2D** - conv. layer 
    - filters - 32
    - kernel_size - 3 x 3
    - activation - 'relu' 
    - input shape - 4D tensor - (n, 56, 56, 1), where (number of examples, img_size, img_size, no_of_channels) 
  - **Max_Pooling** - subsampling layer
    - pool_size - (2, 2)
  - **Dropout** - regularization layer
    - dropout_percentage - 20%

  - **Conv2D** - conv. layer 
    - filters - 32
    - kernel_size - 3 x 3
    - activation - 'relu' 
    - input shape - 4D tensor - (n, 56, 56, 1), where (number of examples, img_size, img_size, no_of_channels) 
  - **Conv2D** - conv. layer 
    - filters - 32
    - kernel_size - 3 x 3
    - activation - 'relu' 
    - input shape - 4D tensor - (n, 56, 56, 1), where (number of examples, img_size, img_size, no_of_channels) 
  - **Max_Pooling** - subsampling layer
    - pool_size - (2, 2)
  - **Dropout** - regularization layer
    - dropout_percentage - 20%
 
  - **Flatten** - flattening input for dense layers input
  - **Dense** - regular dense layer
    - number of neurons - 128
    - activation - 'relu'
   
  - **Dense** - final layer
    - units - number of classes
    - activation - softmax
    
```python
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
```
### 2.2 Hyperparameters tuning

I used random search approach to select set of hyperparameters. 
