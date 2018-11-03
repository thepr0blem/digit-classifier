# CNN Implementation using Keras

The purpose of this project is to develop convolutional neural network for written characters classification. 

## Introduction 

#### Dataset  
Training dataset consist of 30,000 examples of written characters divided into 35 classes - 10 digits and 25 letters. 

There is no class for letter "X". 

Each example is a 56x56 black/white image. Pixels are values 0 or 1. 

Training set is provided in form of numpy arrays with 3 136 columns (with pixel values) and one additional vector with class label. 

#### Content of this document 
1. Data loading, preperation and exploring 
2. CNN modelling
3. Model evaluation

## 1. Loading and exploring the data
