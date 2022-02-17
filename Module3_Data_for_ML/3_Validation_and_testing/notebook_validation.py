#%%
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


X, y = datasets.load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # test_size is a proportion of the data you are going to split

X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.3) # test_size is a proportion of the data you are going to split

print(f"no samples y: {len(y)}")
print(f"no samples y_train: {len(y_train)}")
print(f"no samples y_test: {len(y_test)}")
print(f"no samples y_validation: {len(y_validation)}")

print(f"no samples x_train: {len(X_train)}")

# %%
np.random.seed(2)

models = [
        DecisionTreeRegressor(splitter="random"),
        SVR(),
        LinearRegression()    
        ]

from sklearn import metrics

for model in models:
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_validation_pred = model.predict(X_validation)
    y_test_pred = model.predict(X_test)
    
    train_loss = metrics.mean_squared_error(y_train, y_train_pred)
    validation_loss = metrics.mean_squared_error(y_validation, y_validation_pred)
    test_loss = metrics.mean_squared_error(y_test, y_test_pred)
    
    print(
        f"{model.__class__.__name__}: "
        f"Train loss: {train_loss}"
        f"Validation loss: {validation_loss}"
        f"Test loss: {test_loss}"
        )

# %%
# data leakage example

def calculate_validation_loss(X_train, y_train, X_validation, y_validation):
    model = LinearRegression()

    # Without data leakage, train on train, validate on validation
    model.fit(X_train, y_train)
    y_validation_pred = model.predict(X_validation)
    validation_loss = mean_squared_error(y_validation, y_validation_pred)

    print(f"Validation loss: {validation_loss}")
    
# Without data leakage, train on train, validate on validation
calculate_validation_loss(X_train, y_train, X_validation, y_validation)

# With data leakage, 50 samples from validation added
fail_X_train = np.concatenate((X_train, X_validation[:50]))
fail_y_train = np.concatenate((y_train, y_validation[:50]))

calculate_validation_loss(fail_X_train, fail_y_train, X_validation, y_validation)

# SUMMARY

# Validation set is used to find info about best algorithms, best set of arguments to algoirthms etc.
# Test set is used to check how our algorithm performs on unseen data
# As we tune algorithms according to validation dataset we cannot use it to check performance
# seed is used to ensure reproducibility. Also multiple runs for experiments are good if our code depends on random initialization heavily (we can take mean results of experiments)
# Data leakage is information from validation (or test) leaking into training
# Data leakage leads to falsely good results and should be avoided
# Rule of thumb: imagine you only have training dataset when doing preprocessing. Anything you calculate from it cannot be used in validation or test