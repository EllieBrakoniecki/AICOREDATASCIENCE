#%%
from sklearn import datasets, metrics, preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

######## THE MODEL ###########################
class LinearRegression:
    def __init__(self, optimiser, num_features): 
        # randomly initialise weight and bias
        np.random.seed(2)
        self.w = np.random.randn(num_features) 
        self.b = np.random.randn() 
        self.optimiser = optimiser
        self.losses = []

    def fit(self, X, y):
        for epoch in range(self.optimiser.epochs):   
            y_pred = self.predict(X) 
            new_w, new_b = self.optimiser.step(self.w, self.b, X, y_pred, y) 
            self._update_params(new_w, new_b) 
            loss = LinearRegression.mse_loss(y_pred, y)
            self.losses.append(loss) 
        self.plot_loss()
        print('Final loss value:', loss)
        print('Weight values:', self.w)
        print('Bias values:', self.b)
        score = self.score(y_pred,y)
        print('R2 score:', score)
        
    def predict(self, X): 
        y_pred = X @ self.w + self.b 
        return y_pred
        
    def _update_params(self, new_w, new_b):
        self.w = new_w 
        self.b = new_b 
        
    @staticmethod 
    # define our criterion (loss function)
    def mse_loss(y_pred, y_actual): 
        error = y_pred - y_actual 
        return (error**2).mean()
    
    def score(self, y_pred, y_actual): # R2 goodness of fit 
        y_mean = y_actual.mean()
        SS_tot = ((y_pred - y_mean)**2).sum() 
        SS_res = ((y_actual-y_pred)**2).sum()
        return 1 - (SS_res/SS_tot)
        
    def plot_loss(self):
        plt.figure()
        plt.ylabel('Cost')
        plt.xlabel('Epoch')
        plt.plot(self.losses)
        plt.show()
        
#### THE OPTIMISER #############
# Batch gradient descent optimiser

class GDOptimiser:
    def __init__(self, a, epochs): # a is the learning rate
        self.a = a
        self.epochs = epochs

    def _calc_deriv(self, X, y_pred, y_actual): # see Regression_california file
        error = y_pred - y_actual
        m = error.size
        dLdw = 2/m * X.T.dot(error) # derivative of loss with respect to weights
        dLdb = 2/m * np.sum(error)  # derivative of loss with respect to bias

        return dLdw, dLdb  # return rate of change of loss wrt w and wrt b

    def step(self, w, b, X, y_pred, y_actual):
        dLdw, dLdb = self._calc_deriv(X, y_pred, y_actual)
        new_w = w - self.a * dLdw
        new_b = b - self.a * dLdb
        return new_w, new_b
    
######################################    
#%%
# from aicore.ml import data
# np.random.seed(2)

# # Use `data.split` in order to split the data into train, validation, test
# (X_train, y_train), (X_validation, y_validation), (X_test, y_test) = data.split(
#     datasets.fetch_california_housing(return_X_y=True)
# )
# X_train, X_validation, X_test = data.standardize_multiple(X_train, X_validation, X_test)


np.random.seed(4)
X, y = datasets.fetch_california_housing(return_X_y=True)
scaler = preprocessing.StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)
#%%
np.random.seed(2)
epochs = 1000
a = 0.001
optimiser = GDOptimiser(a=a, epochs=epochs)
model = LinearRegression(optimiser=optimiser, num_features=X_train.shape[1]) 
model.fit(X_train, y_train)




# %%
