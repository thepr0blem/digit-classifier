# CNN Implementation using Keras

The purpose of this project is to develop convolutional neural network for written characters classification. 

## Introduction 


+ What technologies did you use (e.g. Sci-kit, TensorFlow, PyTorch etc.)?
+ How did you validate your model? Provide an estimate of the expected
+ accuracy of the classifier.
+ What techniques did you utilize to improve your method (e.g. data
+ transformation, regularization, data augmentation, hyperparameter tuning,


#### Project structure 

folders
files

#### Dataset overview
Training dataset consist of 30,134 examples of written characters divided into 35 classes - 10 digits and 25 letters. 

There is no class for letter "X". 

Each example is a 56x56 black/white image. Pixels are values 0 or 1. 

Training set is provided in form of numpy arrays with 3,136 columns (with pixel values) and one additional vector with class label. 

**NOTE:** In the original data set there were 36 classes where one of them (class 30) had only one example
          and was overlapped with class 14 (both were letter "N"). Single example from class 30 was moved to
          class 14 and class 35 was renamed to 30.

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
    - padding - 'same'
  - **Conv2D** - conv. layer 
    - filters - 32
    - kernel_size - 3 x 3
    - activation - 'relu' 
    - padding - 'same'
  - **Max_Pooling** - subsampling layer
    - pool_size - (2, 2)
  - **Dropout** - regularization layer
    - dropout_percentage - 20%

  - **Conv2D** - conv. layer 
    - filters - 32
    - kernel_size - 3 x 3
    - activation - 'relu' 
    - padding - 'same'
  - **Conv2D** - conv. layer 
    - filters - 32
    - kernel_size - 3 x 3
    - activation - 'relu' 
    - padding - 'same'
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
def create_model(X, y, it=1, no_of_filters=32, kern_size=5,
                 max_p_size=2, drop_perc_conv=0.2, drop_perc_dense=0.4,
                 dens_size=256, val_split_perc=0.2, no_of_epochs=1,
                 optimizer="adam", random_search=False):
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

    model.add(Conv2D(2 * no_of_filters,
                     kernel_size=(kern_size, kern_size),
                     activation='relu',
                     padding='same'))
    model.add(Conv2D(2 * no_of_filters,
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

    history = model.fit(X,
                        y_train_cat,
                        validation_split=val_split_perc,
                        epochs=no_of_epochs,
                        callbacks=[early_stopping_monitor],
                        batch_size=128)

    history_dict = history.history

    if random_search:

        np.save(r"./models/random_search/hist/history_dict_{}.npy".format(it), history_dict)
        model.save(r"./models/random_search/models/CNN_{}.h5".format(it))

    else:

        np.save(r"./logs/history_dict_{}.npy".format(it), history_dict)
        model.save(r"./models/CNN_model_{}.h5".format(it))

    return history_dict
```
### 2.2 Hyperparameters tuning

I used **random search** approach to select the best set of hyperparameters given time frames nad computational capabilities of my hardware. 

Defining parameters dictionary 
```python
parameters_dct = {"no_of_filters": [8, 16, 32, 48, 64],
                  "kern_size": [3, 4, 5],
                  "max_pool": [2, 3],
                  "dropout_perc": [0.05, 0.1, 0.2, 0.3, 0.4],
                  "dense_size": [64, 128, 192, 256, 320],
                  "optimizers": ["adam", "adamax", "nadam", "RMSProp"]
                  }
```

Defining the function which will perform random search given above set of hyperparameters. 

The function will: 
- use ```create model()``` function from above to train n number of models with randomly selected set of parameters
- it will save all models in ```./models/random_search/models``` 
- it will save set of parameters and training histories for all iterations in ```./models/random_search/params``` and ```.models/random_search/hist``` accordingly 
- only 15% of the whole data set will be used as I am looking only for indication which parameters will be the best instead of the ready model 

```python

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
                                 random_search=True
                                 )

        val_accs_list.append(hist_dict['val_acc'][-1])

    np.save(r"./models/random_search/val_accs_list.npy", val_accs_list)

    return val_accs_list
```

Running the search. 

```python
mod.run_random_search(X_train_cnn, y_train_cnn, parameters_dct, 20)
```

Loading list with accuracies and printing parameters for the "best" combination from random_search. 

```
val_accs_list = np.load(r"./models/random_search/val_accs_list.npy")
```

```
print(np.load(r"./models/random_search/params/params_dict_{}.npy".format(val_accs_list.argmax())))
```

```
{'iteration': 1, 'no_of_filters': 8, 'kern_size': 5, 'max_pool': 2, 'dropout_perc_conv': 0.2, 'dropout_perc_dens': 0.1, 'dense_size': 64, 'optimizer': 'adamax'}
```
Based on above, training model with given parameters on full training data set.

```python
mod.create_model(X_train_cnn, y_train_cnn, it="F", no_of_filters=32, kern_size=5,
                 max_p_size=2, drop_perc_conv=0.2, drop_perc_dense=0.4,
                 dens_size=256, val_split_perc=0.2, no_of_epochs=5,
                 optimizer="adam", random_search=False)
```
Model will be saved as models/CNN_model_F.h5

## 3. Model evaluation
### 3.1 Load model and evaluate on test data set 

```python
def load_saved_model(path):
    """Loads model using keras.load_model() function.

    Args:
        path: Model folder directory.
        filename: Name of the model file (.h5 file type)

    Returns:
        Keras model instance.
    """
    return load_model(path)
```

```
model = mod.load_saved_model(r"./models/CNN_model_F.h5")
```

```
score = mod.test_and_score(model, X_test_cnn, y_test_cat_cnn)
```

```
score
Out[6]: [0.9276472181848217, 0.7765057242804743]
```
Accuracy on test score is XX %. 

### 3.2 Confusion matrix 

Leveraging ```scikit-learn``` modules we can easily build confusion matrix, which will show which classes are the most difficult for the model to distinguish between. 

First, we need to classify test examples and pass the predictions to ```confusion_matrix``` together with true labels.  
```python
y_pred = pred.predict(X_test_cnn)
```
```python
conf_mat = confusion_matrix(y_test_cnn, y_pred)
```
Next, short function for plotting the matrix will be required. I used ```seaborn.heatmap```

```python
def plot_conf_mat(conf_mat, label_list, normalize=False):

    if normalize:

        conf_mat = conf_mat.astype(float) / conf_mat.sum(axis=1)[:, np.newaxis]

    fmt = '.2f' if normalize else 'd'
    sns.heatmap(conf_mat, annot=True, fmt=fmt,
                cmap="Blues", cbar=False, xticklabels=label_list,
                yticklabels=label_list, robust=True)
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
```
Labels list: 
```python
labels_list = [labels[i] for i in range(35)]
```
Plotting: 
```python
vis.plot_conf_mat(conf_mat, labels_list, normalize=False)
```




### 3.3 Accuracy report 

### 3.4 Display exemplary mistakes 

## Conclusions 

Model did pretty well on test set scoring 95% accuracy. 

Ideas which were considered during the development, but were not implemented (indication of potential further are for exploration) 
- data augmentation (via small rotatio, translation and zoom)  
- replacing MaxPooling layers with Conv2D layers with a (2, 2) stride - making subsampling layer also learnable 

### References 
[TEXT TO SHOW](actual URL to navigate)
link 1
link 2
link 3
