#%%

from sklearn.ensemble import RandomForestRegressor

regressors = [
    RandomForestRegressor(n_estimators=10, criterion="mae"),
    RandomForestRegressor(n_estimators=50, min_samples_leaf=2),
    RandomForestRegressor(),
            ]
# Above we have a single classification machine learning method called Random Forest (we will get to how it works in next module).
# What interests us are __init__ parameters we provided (n_estimators, criterion, min_samples_leaf). They are examples of hyperparameters you can set before fitting them to data.
# What can happen after setting them incorrectly?

# our algorithm may under/overfit (more details later)
# it might not converge at all in some cases
# When we do it right (at least more or less) we can observe:

# improved convergence & faster training time
# lower loss & better performance on test data
# %%
# GRID search

# Let's implement a simple Grid Search. To outline what we have to do:

# Create dictionary containing hyperparameters names and their allowed values (as list). Use at most 4 values:
# n_estimators with values [10, 50, 100] (or others, not more than a 1000)
# criterion (check possible values)
# min_samples_leaf with values [2, 1]

import itertools
import typing

def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))


grid = {
    "n_estimators": [10, 50, 100],
    "criterion": ["mse", "mae"],
    "min_samples_leaf": [2, 1],
}

for i, hyperparams in enumerate(grid_search(grid)):
    print(i, hyperparams)
    
# ## Grid search evaluation
# Now that we have our grid creating function we can evaluate 
# RandomForest and see how it works out:    
#%%
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X, y = datasets.load_boston(return_X_y=True)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)

best_hyperparams, best_loss = None, np.inf

for hyperparams in grid_search(grid):
    model = RandomForestRegressor(**hyperparams)
    model.fit(X_train, y_train)

    y_validation_pred = model.predict(X_validation)
    validation_loss = mean_squared_error(y_validation, y_validation_pred)

    print(f"H-Params: {hyperparams} Validation loss: {validation_loss}")
    if validation_loss < best_loss:
        best_loss = validation_loss
        best_hyperparams = hyperparams

print(f"Best loss: {best_loss}")
print(f"Best hyperparameters: {best_hyperparams}")

#%%
from itertools import product

      
def product_dict(options):
    keys = options.keys()
    vals = options.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))
        
options = {"number": [1,2,3], "color": ["orange","blue"] }

for i, option in enumerate(product_dict(options)):
    print(i, option)
# %%
