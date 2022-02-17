#%%

## STOCHASTIC GRADIENT DESCENT 
import matplotlib.pyplot as plt
def plot_loss(losses):
    """Helper function for plotting loss against epoch"""
    plt.figure() # make a figure
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.plot(losses) # plot costs
    plt.show()
# %%
### THE DATA ###

from sklearn import datasets, model_selection
from aicore.ml import data
import pandas as pd
import numpy as np

# Use `data.split` in order to split the data into train, validation, test
(X_train, y_train), (X_validation, y_validation), (X_test, y_test) = data.split(
    datasets.load_boston(return_X_y=True)
)
X_train, X_validation, X_test = data.standardize_multiple(X_train, X_validation, X_test)

# %%
### THE MODEL ###
# Here's the same model we implemented before

class LinearRegression:
    def __init__(self, optimiser, n_features): # initalize parameters 
        self.w = np.random.randn(n_features) ## randomly initialise weight
        self.b = np.random.randn() ## randomly initialise bias
        self.optimiser = optimiser
        
    def predict(self, X): # how do we calculate output from an input in our model?
        ypred = X @ self.w + self.b ## make a prediction using a linear hypothesis
        return ypred # return prediction

    def fit(self, X, y):
        all_costs = [] ## initialise empty list of costs to plot later
        for epoch in range(self.optimiser.epochs): ## for this many complete runs through the dataset    

            # MAKE PREDICTIONS AND UPDATE MODEL
            predictions = self.predict(X) ## make predictions
            print('shape_pred',predictions.shape)
            new_w, new_b = self.optimiser.step(self.w, self.b, X, predictions, y) ## calculate updated params
            self._update_params(new_w, new_b) ## update model weight and bias
            
            # CALCULATE LOSS FOR VISUALISING
            cost = LinearRegression.mse_loss(predictions, y) ## compute loss 
            all_costs.append(cost) ## add cost for this batch of examples to the list of costs (for plotting)

        plot_loss(all_costs)
        print('Final cost:', cost)
        print('Weight values:', self.w)
        print('Bias values:', self.b)

    
    def _update_params(self, new_w, new_b):
        self.w = new_w ## set this instance's weights to the new weight value passed to the function
        self.b = new_b ## do the same for the bias
        
    @staticmethod 
    def mse_loss(y_hat, labels): # define our criterion (loss function)
        errors = y_hat - labels ## calculate errors
        squared_errors = errors ** 2 ## square errors
        mean_squared_error = sum(squared_errors) / len(squared_errors) ## calculate mean 
        return mean_squared_error # return loss
# %%
# THE OPTIMISER _ gradient descent 

import numpy as np

class SGDOptimiser:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs

    def _calc_deriv(self, features, predictions, labels):
        m = len(labels) ## m = number of examples
        diffs = predictions - labels ## calculate errors
        dLdw = 2 * np.sum(features.T * diffs).T / m ## calculate derivative of loss with respect to weights
        dLdb = 2 * np.sum(diffs) / m ## calculate derivative of loss with respect to bias
        return dLdw, dLdb ## return rate of change of loss wrt w and wrt b

    def step(self, w, b, features, predictions, labels):
        dLdw, dLdb = self._calc_deriv(features, predictions, labels)
        new_w = w - self.lr * dLdw
        new_b = b - self.lr * dLdb
        return new_w, new_b
# %%

# PUTTING ALL TOGETHER

num_epochs = 1000
learning_rate = 0.001

optimiser = SGDOptimiser(lr=learning_rate, epochs=num_epochs)
model = LinearRegression(optimiser=optimiser, n_features=X_train.shape[1])
model.fit(X_train, y_train)


#%%
#####  sklearn example #######
# sklearn packs everything we just did above into it's simple LinearRegression API.
from sklearn.linear_model import LinearRegression

linear_regression_model = LinearRegression() ## instantiate the linear regression model


def mse_loss(y_hat, labels): # define our criterion (loss function)
    errors = y_hat - labels ## calculate errors
    squared_errors = errors ** 2 ## square errors
    mean_squared_error = sum(squared_errors) / len(squared_errors) ## calculate mean 
    return mean_squared_error # return loss
    
def calculate_loss(model, X, y):
    return mse_loss(model.predict(X),y)

model = linear_regression_model.fit(X_train, y_train) ## fit the model

print(f"Training loss before fit: {calculate_loss(model, X_train, y_train)}")
print(
    f"Validation loss before fit: {calculate_loss(model, X_validation, y_validation)}"
)
print(f"Test loss before fit: {calculate_loss(model, X_validation, y_validation)}")

# %%
epochs = 10000
model.fit(X_train, y_train)

print(f"Training loss after fit: {calculate_loss(model, X_train, y_train)}")
print(f"Validation loss after fit: {calculate_loss(model, X_validation, y_validation)}")
print(f"Test loss after fit: {calculate_loss(model, X_validation, y_validation)}")

print('final weights:', model.coef_)
print('final bias:', model.intercept_)

#%% 
# FUNCTION TO NORMALISE DATA

def standardize_data(dataset, mean=None, std=None):
    if mean is None and std is None:
        mean, std = np.mean(dataset, axis=0), np.std(
            dataset, axis=0
        )  ## get mean and standard deviation of dataset
    standardized_dataset = (dataset - mean) / std
    return standardized_dataset, (mean, std)

X_train, (mean, std) = standardize_data(X_train)
# %%
