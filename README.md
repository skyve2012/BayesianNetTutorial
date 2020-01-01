# BayesianNetTutorial

This repo includes a tutorial on Bayesian Neural Networks. There are three separate parts for this tutorial.
1. Linear Regression with variable noise
2. Bayesian Neural Network on Gravitational Wave Parameter Estimation (applies to other purposes in general)
3. How to write TensorFLow code for training and evaluation (arbitrary mdoel and dataset)


## Required Packages
Required packges are listed in.
```
requirements.txt
```
It's recommended to use Anaconda to create a virtual environment and install these packages. The codes are tested with Python=2.7.16 envrionment.

## File Description

1. **model.py**: includes model definition and model object which is called during model training
2. **dataset.py**: includes training and testing data generator. Both can be changed based on other usages. Both are fed to TensorFlow Estimator object during training and evaluation.
3. **LinearRegression.ipynb**: includes three different setups of a linear regression model. 
4. **BNN_GW_Estimation.ipynb**: includes codes for loading a trained model and evaluate for gravitational wave parameters
5. **BNN_Train_and_Save.ipynb**: includes training and model saver using TensorFlow Estimator pipeline. This can be changed based on other purposes in accordance with **model.py** and **dataset.py**.



## Reference Link

Here's a reference [link](https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_Regression.ipynb#scrollTo=TLZ97_V4PP-f) for linear regression part


