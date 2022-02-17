#%%
from sklearn import datasets, model_selection

# 15% for validation and test, 70% for train in total
X, y = datasets.fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

X_validation, X_test, y_validation, y_test = model_selection.train_test_split(
    X_test, y_test, test_size=0.5
)
print(X_train.shape, y_train.shape)
# %%
import numpy as np

class LinearRegression:
    def __init__(self, n_features: int): # initalize parameters
        np.random.seed(10)
        self.W = np.random.randn(n_features, 1) ## randomly initialise weight
        self.b = np.random.randn(1) ## randomly initialise bias
        
    def __call__(self, X): # how do we calculate output from an input in our model?
        ypred = np.dot(X, self.W) + self.b
        return ypred # return prediction
    
    def update_params(self, W, b):
        self.W = W ## set this instance's weights to the new weight value passed to the function
        self.b = b ## do the same for the bias
        
# linear_reg = LinearRegression(5)
# np.random.seed(2)
# X = np.random.randn(15,5)
# pred = linear_reg(X)
# pred.shape
# %%
model = LinearRegression(n_features=8)  # instantiate our linear model
y_pred = model(X_test)  # make prediction on data
print("Predictions:\n", y_pred[:10]) # print first 10 predictions
# %%
import matplotlib.pyplot as plt

def plot_predictions(y_pred, y_true):
    samples = len(y_pred)
    plt.figure()
    plt.scatter(np.arange(samples), y_pred, c='r', label='predictions')
    plt.scatter(np.arange(samples), y_true, c='b', label='true labels', marker='x')
    plt.legend()
    plt.xlabel('Sample numbers')
    plt.ylabel('Values')
    plt.show()
# %%
plot_predictions(y_pred[:10], y_test[:10])
# %%

def mean_squared_error(y_pred, y_true):  # define our criterion (loss function)
    errors = y_pred - y_true  ## calculate errors
    squared_errors = errors ** 2  ## square errors
    return np.mean(squared_errors)

cost = mean_squared_error(y_pred, y_train)
print(cost)
# %%
# minimise MSE: 
def minimize_loss(X_train, y_train):
    X_with_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    optimal_w = np.matmul(
        np.linalg.inv(np.matmul(X_with_bias.T, X_with_bias)),
        np.matmul(X_with_bias.T, y_train),
    )
    return optimal_w[1:], optimal_w[0]


weights, bias = minimize_loss(X_train, y_train)
print(weights, bias)
# %%
model.update_params(weights, bias)
y_pred = model(X_train)
cost = mean_squared_error(y_pred, y_train)
print(cost)
# %%

plot_predictions(y_pred[:10], y_train[:10])
# %%
# sklearn comes with some common machine learning models that you can use out of the box, with a simple to use API.
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.fetch_california_housing(return_X_y=True)

model = linear_model.LinearRegression()

#How to use sklearn's API of models
# sklearn machine learning algorithms are objects which usually follow this general convention:

# __init__(*args, **kwargs) - here you setup your algorithm (as seen above).
# fit(X, [y]) - train the model on X (features) and y (targets). In case of unsupervised algorithms there is no y
# predict(X) - pass data (previously unseen) to algorithm after fit was called. This gives us predictions (y_pred).
model.fit(X, y)
y_pred = model.predict(X)
print(y_pred[:5], "\n", y[:5])
# %%
plot_predictions(y_pred[:10], y[:10])
#%%
from sklearn import metrics

metrics.mean_squared_error(y, y_pred)
# %%
# Model persistance
# Training (fitting) process is often quite expensive (in time and or compute cost), while what we are after is the ability to predict on unseen data.
# We see our model works okay and we would like to save it for later use without the need to train on the data again.
# Model persistence means saving your machine learning algorithm currently held in RAM (Random Access Memory) to a storage (usually hard drive) from which it can be reinstantiated at any point in time
# As per usual it's simple with sklearn:

import joblib

joblib.dump(model, "model.joblib")  
# %%
# Downsides
# As sklearn is very high level it doesn't require much knowledge to use as is. But we have to know more in order to do machine learning well. What is missing here:

# Why and what for? There are many more ways (and way more correct) to do machine learning
# Knowledge of machine learning algorithms; we have to know which one to choose for which kind of problems
# Knowledge of possible pitfalls; machine learning can easily go wrong. We have to know more about it in order to improve our model's performance
# In-depth knowledge of the ideas; often it might be a good idea to implement major ideas on your own
# We will do all of the above, but hopefully you can see how easy and definitely not scary it can be.

# sklearn tips
# Always try easiest solution first. Create a weak baseline algorithm and check how it performs. Do not go straight to the most complicated ones! It is called Occam's Razor in philosophy and machine learning also
# Some algorithms have attributes you might be interested in. Those are usually suffixed by _ underscore, for example my_algorithm.interesting_attribute_
# Some __init__ functions have a lot of possible arguments. Each of them influences how the algorithm works. But which are the most important and have the most influence? In sklearn those arguments come in order from most influential to least
# Many sklearn algorithms provide n_jobs argument, which parallelizes fit, predict and other functions. You can use n_jobs=-1 to use as many processes as there are virtual cores (it is often a reasonable amount), which improves performance tremendously.
# Use idiomatic sklearn - search the documentation, use pipelines if possible

# Summary
# linear regression is "hello world" basic machine learning model
# linear regression updates it's weight vector and bias in order to improve on the task
# this update can be carried out via analytically calculated formula
# the MSE loss is appropriate for many regression problems and is the most common loss function for this task