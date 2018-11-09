# CNN Implementation using Keras

The purpose of this project is to develop convolutional neural network for written characters classification. 

### Content of this document 
1. Introduction
2. Data loading, exploring and preprocessing
3. CNN model architecture selection, initialization and training
4. Model evaluation

## 1. Introduction 

### 1.1 Technologies and techniques used: 

#### Model architecture 
```Keras``` library (framework based on ```Tensorflow```) 

#### Validation:
Tools provided in ```scikit-learn``` library:
- random division of the sample on training and testing sets
- confusion matrix 

#### Techniques for accuracy improvement:
Estimated accuracy of the classifier: 95.2%. Based on model performance calculated from testing set accuracy. 
Techniques: 
- regularization (via Dropout) 
- hyperparameter tuning (via Random Search) 
- early stopping 
- learning rate reduction (via ReduceLROnPlateau) 
- gradient descent optimization ("adam", "adamax", "nadam", "RMSProp")
- image augmentation (rotation and shift) - tested, but not used

### 1.2 Project structure 

```
├── data                    # Data sets
├── logs                    # Training history logs 
├── models                  # Trained models 
├── pics                    # Pictures/visualizations 
├── src                     # Source files 
├── workflow.py             # Code for workflow as presented in README
├── predict.py              # Function for new data classification 
├── requirments.txt         # Required libraries
└── README.md                 
```

### 1.3 Dataset overview

**NOTE:** In the original data set there were 36 classes and one of them (class #30) had only one example.
          This class was overlapping with class #14 (both were letter "N"), single example was renamed #30 -> #14. 

Observations: 
- training set is provided in form of numpy arrays with 3,136 columns (with pixel values) and one additional vector with class labels
- training dataset consist of 30,134 examples of written characters divided into 35 classes - 10 digits and 25 letters
- the characters are centered and aligned in terms of the size
- the classes are not in order (letters and digits are mixed) 
- there is no data/class for letter "X" 
- each example is a 56x56 image unfolded to 1x3136 vector (data need to be reshaped before feeding into CNN model) 
- pixel values are binary (0 - black / 1 - white) 

### 1.4 How to USE

Required technologies are listed in ```requirments.txt``` file.

#### 1.4.1 ```workflow.py```

To go through all te steps described in this document please use ```workflow.py``` script 

Imports used in the script with all dependencies 

```python
import numpy as np
import pickle
import seaborn as sns
import random as rd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.models import load_model

from src import modelling as mod
from src import visualization as vis
import predict as pred
```

#### 1.4.2 ```predict.py```

To leverage already trained model to make your own prediction, use ```predict.py``` 

```python
def predict(input_data):
    """This functions classifies given input 

    Args:
        input_data: (n x 3136) array, where n - # of examples
        
    Returns:
        output_data: (n x 1) array with class labels 
    """
    model_in = load_model(r"./models/CNN_1.h5")

    prediction = model_in.predict(input_data)

    output_data = prediction.argmax(axis=1).reshape(len(prediction), 1)

    return output_data
```

## 2. Loading and exploring the data

### 2.1 Loading the data
```python
data_dir = r'./data/train_fix.pkl'

with open(data_dir, 'rb') as f:
    data = pickle.load(f)

X, y = data
```
Label dictionary 
```python
labels = {0: "6", 1: "P", 2: "O", 3: "V", 4: "W", 5: "3", 6: "A", 
          7: "8", 8: "T", 9: "I", 10: "0", 11: "9", 12: "H", 13: "R", 
          14: "N", 15: "7", 16: "K", 17: "L", 18: "G", 19: "4", 20: "Y", 
          21: "C", 22: "E", 23: "J", 24: "5", 25: "1", 26: "S", 27: "2", 
          28: "F", 29: "Z", 31: "Q", 32: "M", 33: "B", 34: "D", 35: "U"}
```  
### 2.2 Exploring data - samples
Plotting classes distribution
```python
y_vec = y.reshape(y.shape[0], )

y_classes = sorted([labels[i] for i in y_vec])

sns.set(style="darkgrid")
count_plot = sns.countplot(y_classes, palette="Blues_d")

plt.show()
```
![Classes](https://github.com/thepr0blem/task/blob/master/pics/data_viz.png) 

**Observation**: Classes are not balanced - class "N" consist of more than 3,000 examples, where some of the classes have less than 300 examples. 

Defining function to plotting randomly selected, exemplary pictures:
```python
def plot_samples(X, y):
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            n = rd.randint(0, X.shape[1])
            axarr[i, j].imshow(X[n].reshape(56, 56), cmap='gray')
            axarr[i, j].axis('off')
            axarr[i, j].set_title(labels[y[n][0]])

```
```
vis.plot_samples(X, y)
```
![Classes](https://github.com/thepr0blem/task/blob/master/pics/samples.png) 

### 2.3 Preprocessing the data
#### 2.3.1 Reshaping 

```python
n_cols = X.shape[1]
img_size = int(np.sqrt(n_cols))
no_of_classes = len(np.unique(y, return_counts=True)[0])

X_cnn = X.reshape(X.shape[0], img_size, img_size, 1)
```

#### 2.3.2 Split into training and testing set 
The data has been split into two data sets in 80:20 proportion.  
```python
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y, test_size=0.2, random_state=42)
```

#### 2.3.3 Label encoding 

```python
y_train_cat_cnn = to_categorical(y_train_cnn)
y_test_cat_cnn = to_categorical(y_test_cnn)
```
      
## 3. Building CNN 
### 3.1 Defining CNN architecture

To implement convolutional neural network I used **Keras** API (which is user friendly framework built on top of Tensorflow). I used Sequential model which is ordered hierarchy of layers. The architecture has been chosen based on research and articles listed in references ([1] in this case). 

Alternative approach that could be taken: adding additional functionality to ```run_random_search``` function which would consider different numbers of sets of layers. 

Layers for final CNN are ordered as follows (selection of hyperparameters is presented in the following steps):

  - **Conv2D** - conv. layer 
    - filters - 40
    - kernel_size - 5 x 5
    - activation - 'relu' 
    - input shape - 4D tensor - (n, 56, 56, 1), where (number of examples, img_size, img_size, no_of_channels) 
    - padding - 'same'
  - **Conv2D** - conv. layer 
    - filters - 40
    - kernel_size - 5 x 5
    - activation - 'relu' 
    - padding - 'same'
  - **Max_Pooling** - subsampling layer
    - pool_size - (3, 3)
  - **Dropout** - regularization layer
    - dropout_percentage - 30%

  - **Conv2D** - conv. layer 
    - filters - 40
    - kernel_size - 5 x 5
    - activation - 'relu' 
    - padding - 'same'
  - **Conv2D** - conv. layer 
    - filters - 40
    - kernel_size - 5 x 5
    - activation - 'relu' 
    - padding - 'same'
  - **Max_Pooling** - subsampling layer
    - pool_size - (3, 3)
  - **Dropout** - regularization layer
    - dropout_percentage - 30%
 
  - **Flatten** - flattening input for dense layers input
  - **Dense** - regular dense layer
    - number of neurons - 512
    - activation - 'relu'
  - **Dropout** - regularization layer
    - dropout_percentage - 30%
   
  - **Dense** - final layer
    - units - number of classes
    - activation - 'softmax'
    
```python
def create_model(X, y, it=1, no_of_filters=32, kern_size=3,
                 max_p_size=3, drop_perc_conv=0.3, drop_perc_dense=0.2,
                 dens_size=128, val_split_perc=0.1, no_of_epochs=5,
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

    model.add(Dense(36, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping_monitor = EarlyStopping(patience=5)
    rlrop = ReduceLROnPlateau(monitor='val_acc', factor=0.5, 
                              patience=3, verbose=1, min_lr=0.00001)

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
        model.save(r"./models/CNN_FF_{}.h5".format(it))

    return history_dict
```
### 3.2 Hyperparameters tuning

I used **random search** approach to select the best set of hyperparameters given time frames nad computational capabilities of my hardware. 

Defining parameters dictionary 
```python
parameters_dct = {"no_of_filters": [8, 16, 24, 32, 40, 48, 56, 64],
                  "kern_size": [3, 4, 5, 6, 7],
                  "max_pool": [2, 3, 4],
                  "dropout_perc": [0.1, 0.2, 0.3, 0.4, 0.5],
                  "dense_size": [64, 128, 192, 256, 384, 512],
                  "optimizers": ["adam", "adamax", "nadam", "RMSProp"],
                  "batch_size": [16, 32, 32, 48, 64, 96, 128]
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
        X: training data
        y: labels for training data
        params: Dictionary with hyperparameters for CNN random search.
        no_of_searches: How many times random search is executed.

    Returns:
        List of accuracies performed on validation set for each iteration.
    """
    val_accs_list = []

    for i in range(no_of_searches):
        # Creating a dict for each iteration of randomly selected parameters

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
        
        # Training model using selected parameters 
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
mod.run_random_search(X_train_cnn, y_train_cnn, parameters_dct, 70)
```

Loading list with accuracies and printing "top 3" sets of parameters from random_search. 

```python
val_accs_list = np.load(r"./models/random_search/val_accs_list.npy")
```

```python
top_three_indices = np.argsort(val_accs_list)[::-1][:3]

for i in top_three_indices:

    print(np.load(r"./models/random_search/params/params_dict_{}.npy".format(i)))
```
```
{'iteration': 69, 'no_of_filters': 24, 'kern_size': 3, 'max_pool': 3, 'dropout_perc_conv': 0.1, 'dropout_perc_dens': 0.4, 'dense_size': 384, 'optimizer': 'adam', 'batch_size': 24}
{'iteration': 11, 'no_of_filters': 56, 'kern_size': 6, 'max_pool': 3, 'dropout_perc_conv': 0.3, 'dropout_perc_dens': 0.4, 'dense_size': 384, 'optimizer': 'adam', 'batch_size': 24}
{'iteration': 56, 'no_of_filters': 40, 'kern_size': 5, 'max_pool': 3, 'dropout_perc_conv': 0.3, 'dropout_perc_dens': 0.3, 'dense_size': 512, 'optimizer': 'adam', 'batch_size': 24}
```

Training three models based on top-3 sets of parameters: 
```python
mod.create_model(X_train_cnn, y_train_cnn, it="1", no_of_filters=24, kern_size=3,
                 max_p_size=3, drop_perc_conv=0.1, drop_perc_dense=0.4,
                 dens_size=384, val_split_perc=0.1, no_of_epochs=40,
                 optimizer="adam", random_search=False, batch_size=24)

mod.create_model(X_train_cnn, y_train_cnn, it="2", no_of_filters=56, kern_size=6,
                 max_p_size=3, drop_perc_conv=0.3, drop_perc_dense=0.4,
                 dens_size=384, val_split_perc=0.1, no_of_epochs=40,
                 optimizer="adam", random_search=False, batch_size=24)

mod.create_model(X_train_cnn, y_train_cnn, it="3", no_of_filters=40, kern_size=5,
                 max_p_size=3, drop_perc_conv=0.3, drop_perc_dense=0.3,
                 dens_size=512, val_split_perc=0.1, no_of_epochs=40,
                 optimizer="adam", random_search=False, batch_size=24)
```

```
{'iteration': 13, 'no_of_filters': 64, 'kern_size': 7, 'max_pool': 3, 'dropout_perc_conv': 0.3, 'dropout_perc_dens': 0.2, 'dense_size': 128, 'optimizer': 'nadam'}
```

## 4. Model evaluation
### 4.1 Load model and evaluate on test data set 

```python
model_1 = keras.models.load_model(r"./models/CNN_FF_1.h5")
model_2 = keras.models.load_model(r"./models/CNN_FF_2.h5")
model_3 = keras.models.load_model(r"./models/CNN_FF_3.h5")
```

```python
score_1 = model_1.evaluate(X_test_cnn, y_test_cat_cnn, batch_size=64)
score_2 = model_2.evaluate(X_test_cnn, y_test_cat_cnn, batch_size=64)
score_3 = model_3.evaluate(X_test_cnn, y_test_cat_cnn, batch_size=64)
```

```python
print("Model_1: val_acc - {}".format(np.round(score_1[1] * 100, 2)),
      "\nModel_2: val_acc - {}".format(np.round(score_2[1] * 100, 2)),
      "\nModel_3: val_acc - {}".format(np.round(score_3[1] * 100, 2)))
```
```
Model_1: val_acc - 94.72% 
Model_2: val_acc - 95.16% 
Model_3: val_acc - 95.19%
```

Highest accuracy on test set has been identified for model_3 which is considered as final from now. 
Its accuracy is estimated on 95.2%.

### 4.2 Confusion matrix 

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
![Conf_mat](https://github.com/thepr0blem/task/blob/master/pics/conf_mat_new.png) 

### 4.3 Display exemplary mistakes 

Define ```display_errors()``` function. 

```python
def display_errors(X, y_true, y_pred, labels):
    """This function shows 9 wrongly classified images (randomly chosen)
    with their predicted and real labels """

    errors = (y_true - y_pred != 0).reshape(len(y_pred), )

    X_errors = X[errors]
    y_true_errors = y_true[errors]
    y_pred_errors = y_pred[errors]

    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
    for i in range(3):
        for j in range(3):
            n = rd.randint(0, len(X_errors))
            ax[i, j].imshow(X_errors[n].reshape(56, 56), cmap='gray')
            ax[i, j].set_title("Predicted label :{}\nTrue label :{}"
                               .format(labels[y_pred_errors[n][0]], labels[y_true_errors[n][0]]))
```

![Conf_mat](https://github.com/thepr0blem/task/blob/master/pics/sample_errors.png) 

## Summary 

- estimated model accuracy - 95.2% 
- based on insightful view presented in confusion matrix, we can conclude that the model misclassifies characters with similar shape Examples:  
  - "o" vs "0" 
  - "i" vs "1"
  - "z' vs "2"
  - "v" vs "u" 
- errors made by classifier are easier to understand if we take a look at exemplary errors in section 4.3. Some of those probably could be also misclassified by human eye
- during the development process data augmentation (via small 10 degree rotation and 0.1 relative position translation) was also considered and tested. However, the same model had 93.2% accuracy on test set therefore the solution was not adapted 

### References 
[1] https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist

[2] https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6/notebook

[3] https://www.kaggle.com/gpreda/cnn-with-tensorflow-keras-for-fashion-mnist

[4] https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist

[5] https://www.kaggle.com/diegosch/classifier-evaluation-using-confusion-matrix
