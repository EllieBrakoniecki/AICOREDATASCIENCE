#%%
from sklearn import datasets, preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import itertools
import typing

#%%
########################################################################
# California housing using Sklearn #

np.random.seed(4)

X, y = datasets.fetch_california_housing(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_train)

print('coeffs:',model.coef_)
print('bias:',model.intercept_)
print('mse:',mean_squared_error(y_train, y_pred))
print('r2_score:',r2_score(y_train, y_pred)) # Coeff of determination -1 is best score
#%%
def plot_predictions(y_pred, y_actual):
    # samples = len(y_pred)
    samples = 20
    plt.figure()
    plt.scatter(np.arange(samples), y_pred[:20], c='r', label='predictions')
    plt.scatter(np.arange(samples), y_actual[:20], c='b', label='true labels', marker='x')
    plt.legend()
    plt.xlabel('Sample numbers')
    plt.ylabel('Values')
    plt.show()
    
plot_predictions(y_pred, y_train)
# %%
############################################################
# california housing using own implementation of gradient descent (rough version see below for class etc)
import numpy as np 
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(4)

X, y = datasets.fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)

X_train_b = np.c_[np.ones((X_train.shape[0],1)), X_train] 

alpha = 0.1
epochs = 1000

# alpha = 0.001
# epochs = 16000
m = y_train.shape[0]
w_b = np.random.randn(X_train_b.shape[1]) # all params incl b 
MSE = []
for i in range(epochs):
    y_pred = X_train_b.dot(w_b)
    error = y_pred - y_train  
    gradients = 2/m * X_train_b.T.dot(error)
    w_b = w_b - (alpha * gradients)
    MSE.append((error**2).mean())
    
y_mean = y_train.mean()
y_pred = X_train_b.dot(w_b)
SS_tot = ((y_train - y_mean)**2).sum() 
SS_res = ((y_train - y_pred)**2).sum()
R2 = 1 - (SS_res/SS_tot)
print(w_b)
print(R2)
print(MSE[-1])

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.plot(MSE)
plt.show()

# %%
###############################################
#Putting the implementation of gradient descent above into a class:
from sklearn import datasets, preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

class LinearRegression:
    def __init__(self, optimiser, num_features): 
        self.theta = np.random.randn(num_features+1) # theta is w and b 
        self.optimiser = optimiser
        self.losses = [] # losses for each epoch

    def fit(self, X, y):
        for epoch in range(self.optimiser.epochs):   
            y_pred = self.predict(X) 
            error = y_pred - y 
            new_theta = self.optimiser.step(self.theta, X, error) 
            self.theta = new_theta             
            loss = (error**2).mean() # MSE
            self.losses.append(loss) 
        self.show_info_and_plot_loss(y_pred, y)
            
    def predict(self, X): 
        X_b = np.c_[np.ones((X.shape[0],1)), X] # add first col of ones to account for b 
        return X_b.dot(self.theta)
    
    @staticmethod
    def MSE(y_pred, y_actual):
        error = y_pred - y_actual 
        return (error**2).mean()
            
    def get_losses(self):
        return self.losses
                            
    def plot_losses(self):
        plt.figure()
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.plot(self.losses)
        plt.show()  
        
    def show_info_and_plot_loss(self, y_pred, y_actual):              
        self.plot_losses()
        print('Final loss value(MSE):', self.losses[-1])
        print('Weight values:', self.theta[1:])
        print('Bias values:', self.theta[0])
        score = self.score(y_pred,y_actual)
        print('R2 score:', score)
    
    def score(self, y_pred, y_actual): # R2 goodness of fit 
        y_mean = y_actual.mean()
        SS_tot = ((y_actual - y_mean)**2).sum() 
        SS_res = ((y_actual-y_pred)**2).sum()
        return 1 - (SS_res/SS_tot)
    
# Batch gradient descent optimiser
class GDOptimiser:
    def __init__(self, alpha, epochs): # alpha is the learning rate
        self.alpha = alpha
        self.epochs = epochs

    def _calc_deriv(self, X, error):
        X_b = np.c_[np.ones((X.shape[0],1)), X] # add first col of ones to account for b 
        m = error.shape[0] # length of the error vector
        gradients = 2/m * X_b.T.dot(error) # vector of derivatives of loss with respect to each parameter (bias and weights)
        return gradients
    
    def step(self, theta, X, error):
        gradients = self._calc_deriv(X, error)
        return theta - (self.alpha * gradients) # return updated gradient vector
        
        
#%%
##############################################
np.random.seed(4)
X, y = datasets.fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)

alpha = 0.1
epochs = 1000
optimiser = GDOptimiser(alpha, epochs)
model = LinearRegression(optimiser, X_train.shape[1]) 
model.fit(X_train, y_train)
# %%
############################################################
# Validation practical 

np.random.seed(4)
X, y = datasets.fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.fit_transform(X_validation)

alpha = 0.1
epochs = 1000
optimiser = GDOptimiser(alpha, epochs)
model = LinearRegression(optimiser, X_train.shape[1]) 
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_validation_pred = model.predict(X_validation)

train_loss = LinearRegression.MSE(y_train_pred, y_train)
validation_loss = LinearRegression.MSE(y_validation_pred, y_validation)

print(f"Train Loss: {train_loss} | Validation Loss: {validation_loss}")
     
# %%
# Hyperparameters, Grid Search & K-Fold Cross Validation practical