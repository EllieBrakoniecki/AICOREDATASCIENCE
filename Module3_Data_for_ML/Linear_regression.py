#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics, preprocessing
from sklearn.model_selection import train_test_split
import itertools
import typing

class LinearRegression():
    def __init__(self, n_features, optimiser):
        np.random.seed(2)
        self.w = np.random.randn(n_features)
        self.b = np.random.randn()
        self.optimiser = optimiser
    
    def fit(self, X, y):
        '''
        Fit model to data
        '''
        losses = []
        for epoch in range(self.optimiser.epochs):
            y_pred = self.predict(X)
            new_w, new_b = self.optimiser.step(self.w, self.b, X, y_pred, y)
            self._update_params(new_w, new_b)
            losses.append(LinearRegression.mse_loss(y_pred, y))
        
        LinearRegression.plot_loss(losses)
        print('Final cost:', losses[-1])
        print('Weight values:', self.w)
        print('Bias values:', self.b)

    def predict(self, X):
        '''
        Calculate prediction
        '''
        y_pred = np.dot(X, self.w) + self.b
        return y_pred
    
    @staticmethod
    def mse_loss(y_pred, y_true):
        '''
        Calculate mean squared error
        '''
        m = y_pred.size
        errors = y_pred - y_true
        mse = 1/m * np.dot(errors.T, errors)
        return mse

    @staticmethod
    def plot_loss(losses):
        '''
        Plot losses
        '''
        plt.figure()
        plt.ylabel('Cost')
        plt.xlabel('Epoch')
        plt.plot(losses)
        plt.show()

    def _update_params(self, w, b):
        '''
        Update parameters
        '''
        self.w = w
        self.b = b
        return w, b

    def score(self, y_pred, y_true):
        '''
        Calculate R2 score
        '''
        u = np.dot((y_pred - y_true).T, (y_pred - y_true))
        y_true_mean = np.full(y_true.shape, np.mean(y_true))
        v = np.dot((y_true_mean - y_true).T, (y_true_mean - y_true))
        R2 = 1 - u/v
        return R2

class SGDOptimiser:
    def __init__(self, alpha, epochs):
        self.alpha = alpha
        self.epochs = epochs

    def _calc_deriv(self, X, y_pred, y_true):
        '''
        Calculate derivate of mean square error(loss) with respect to parameters
        '''
        m = y_pred.size
        errors = y_pred - y_true
        dLdw = 2/m * np.sum(X.T * errors).T
        print('dLdw',dLdw)
        dLdb = 2/m * np.sum(errors)
        print('dLdb',dLdb)
        return dLdw, dLdb

    def step(self, w, b, X, y_pred, y_true):
        '''
        Calculate updated paramters to decrease mean square error
        '''
        dLdw, dLdb = self._calc_deriv(X, y_pred, y_true)
        new_w = w - self.alpha * dLdw
        new_b = b - self.alpha * dLdb
        return new_w, new_b

class DataLoader:
    def __init__(self, X, y):
        idx = np.random.permutation(X.shape[0])
        self.X = X[idx]
        self.y = y[idx]

    def yield_data(self, n):
        X_yield = self.X[0:n+1]
        y_yield = self.y[0:n+1]
        self.X = self.X[n+1:]
        self.y = self.y[n+1:]
        return X_yield, y_yield

    def add_data(self, X_new, y_new):
        self.X = np.append(X, X_new)
        self.y = np.append(y, y_new)

#%%
np.random.seed(2)
X, y = datasets.fetch_california_housing(return_X_y=True)
scaler = preprocessing.StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

np.random.seed(2)
epochs = 1000
a = 0.001
optimiser = SGDOptimiser(alpha=a, epochs=epochs)
model = LinearRegression(optimiser=optimiser, n_features=X_train.shape[1]) 
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
score = model.score(y_pred,y_train)
print(score)
# epochs = 2000
# # alpha = 3 * 1e-4
# # batch_size = X_train.shape[0] // 4

# def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
#     keys, values = zip(*hyperparameters.items())
#     yield from (dict(zip(keys, v)) for v in itertools.product(*values))

# grid = {
#     "alpha": [1e-3, 3*1e-4, 1e-4],
#     "batch_size": [X_train.shape[0] // 5, X_train.shape[0] // 3, X_train.shape[0]],
#     }

# best_hyperparams, best_loss, best_model = None, np.inf, None

# for hyperparams in grid_search(grid):
#     alpha = hyperparams['alpha']
#     batch_size = hyperparams['batch_size']

#     optimiser = SGDOptimiser(alpha, epochs)
#     model = LinearRegression(optimiser=optimiser, n_features=X_train.shape[1])

#     data_loader = DataLoader(X_train, y_train)
#     while True:
#         X_t, y_t = data_loader.yield_data(batch_size)
#         if X_t.shape[0] == 0:
#             break
#         print(X_t.shape)
#         model.fit(X_t, y_t)
#     loss = model.mse_loss(model.predict(X_val), y_val)
#     if loss < best_loss:
#         best_loss = loss
#         best_hyperparams = hyperparams
#         best_model = model

# print("Best model params: ", best_model.w, best_model.b)
# print("Best hyperparameters: ", best_hyperparams)
# print("Best loss: ", best_loss)
# print("Train set mse: ", best_model.mse_loss(model.predict(X_train), y_train))
# print("Validation set mse: ", best_model.mse_loss(model.predict(X_val), y_val))
# print("Test set mse: ", best_model.mse_loss(model.predict(X_test), y_test))


# def k_fold(dataset, n_splits: int = 5):
#     chunks = np.array_split(dataset, n_splits)
#     for i in range(n_splits):
#         training = chunks[:i] + chunks[i + 1 :]
#         validation = chunks[i]
#         yield np.concatenate(training), validation
# print("KFOLD")
# loss = 0
# n_splits = 5
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# for (X_train, X_val), (y_train, y_val) in zip(k_fold(X_train, n_splits), k_fold(y_train, n_splits)):
#     scaler = preprocessing.StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_val = scaler.transform(X_val)
#     optimiser = SGDOptimiser(best_hyperparams['alpha'], epochs)
#     model = LinearRegression(optimiser=optimiser, n_features=X_train.shape[1])
#     data_loader = DataLoader(X_train, y_train)
#     while True:
#         X_t, y_t = data_loader.yield_data(best_hyperparams['batch_size'])
#         if X_t.shape[0] == 0:
#             break
#         print(X_t.shape)
#         model.fit(X_t, y_t)
#     y_validation_pred = model.predict(X_val)
#     fold_loss = model.mse_loss(y_val, y_validation_pred)
#     loss += fold_loss
#     print(f"Fold loss: {fold_loss}")
# print(f"K-Fold estimated loss: {loss / n_splits}")
# %%

# %%
