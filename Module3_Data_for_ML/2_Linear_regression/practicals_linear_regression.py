# Practical 1

# Load in the California house pricing data and unpack the features and labels
# Import a linear regression model from sklearn
# Fit the model
# Create a fake house's features and predict it's price
# Compute the score of the model on the training data
#%%
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

#%%
# X, y = datasets.fetch_california_housing(return_X_y=True)
housing_data = datasets.fetch_california_housing()
X = housing_data.data
y = housing_data.target
print(X.shape)
print(y.shape)
print(housing_data.feature_names)
print(housing_data.DESCR)

#%%
model = linear_model.LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # test_size is a proportion of the data you are going to split

X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.3) # test_size is a proportion of the data you are going to split

print(len(y))
print(f"num samples y_train: {len(y_train)}")
print(f"num samples y_test: {len(y_test)}")
print(f"num samples y_validation: {len(y_validation)}")

print(len(X))
print(f"num samples X_train: {len(X_train)}")
print(f"num samples X_test: {len(X_test)}")
print(f"num samples X_validation: {len(X_validation)}")

# %%
np.random.seed(2)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_validation_pred = model.predict(X_validation)
y_test_pred = model.predict(X_test)

train_loss = metrics.mean_squared_error(y_train, y_train_pred)
validation_loss = metrics.mean_squared_error(y_validation, y_validation_pred)
test_loss = metrics.mean_squared_error(y_test, y_test_pred)

train_score = model.score(X_train, y_train)
validation_score = model.score(X_validation, y_validation)
test_score = model.score(X_test, y_test)

print(
    f"{model.__class__.__name__}: "
    f"Train score: {train_score}"
    f"Validation score: {validation_score}"
    f"Test score: {test_score}"
    )
#%%
X_fake_house = np.array([[ 6.92710000e+00,  1.90000000e+01,  5.53584906e+00,
         9.89245283e-01,  1.72300000e+03,  3.63407547e+00,
         2.98100000e+01, -1.37660000e+02]])
y_fake_house_pred = model.predict(X_fake_house)
print(y_fake_house_pred)
# %%

# # %%
# PRACTICAL: Access the sklearn parameters
# Fit a linear regression model to the California housing dataset
# Take a look at the docs, and figure out how to print the weights and bias that this model has learnt for the dataset
# Take a look at the docs for the dataset, and
# Discuss: what does this tell you about the importance of each feature?

print(model.coef_)
print(model.intercept_)
print(housing_data.feature_names)
# %%
# PRACTICAL: Visualise the sklearn parameters
# Take a single feature of the housing dataset
# Scatter plot it against the label in an X-Y graph
# Fit a model to that feature
# Plot your predictions on top (as a line, not a scatter)
# Discuss: what do you expect the weight and bias values to be?
# Access the weight and bias from the model and print them
# Were your expectations correct?
from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing(as_frame=True)
california_housing.frame.head()
# %%
import matplotlib.pyplot as plt
california_housing.frame['MedInc'].describe()
california_housing.frame['MedHouseVal'].describe()
subset_df = california_housing.frame[['MedInc','MedHouseVal']]  

import matplotlib.pyplot as plt

subset_df.hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.subplots_adjust(hspace=0.7, wspace=0.4)

subset_df.plot(kind='scatter', x='MedInc', y='MedHouseVal', alpha=0.1)

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
X = subset_df[['MedInc']]
y = subset_df['MedHouseVal']
model.fit(X,y)
y_pred = model.predict(X)
print(model.coef_)
print(model.intercept_)
print(mean_squared_error(y, y_pred))
print(r2_score(X, y)) # Coeff of determination -1 is best score

plt.scatter(X, y, color="black", alpha=0.1)
plt.plot(X, y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()


# %%
# Draw the loss function
# Fit a linear regression model to predict the house prices from one column of the California house price dataset
# Access the weight and bias from the model
# One by one, set the models' weight parameter equal to the value in a range of values 
# from 10 below and 10 above the found weight and calculate the mean square error 
# (hint: there's an sklearn tool for computing the MSE)
# Plot the loss agains the parameter value
# Discuss: does it look how you expect?
california_housing.frame.head()
california_housing.frame['AveRooms'].describe()
X = california_housing.frame[['AveRooms']]
y = california_housing.frame['MedHouseVal']
model.fit(X,y)
y_pred = model.predict(X)
weight = model.coef_
print(weight)
bias = model.intercept_
print(bias)
mse = mean_squared_error(y, y_pred)
print(mse)
r2_score = model.score(X, y)
print(r2_score)

plt.scatter(X, y, color="black", alpha=0.1)
plt.plot(X, y_pred, color="blue", linewidth=3)
plt.xlabel('AveRooms')
plt.ylabel('MedianHouseVal')
plt.show()
#%%
# One by one, set the models' weight parameter equal to the value in a range of values 
# from 10 below and 10 above the found weight and calculate the mean square error 
# (hint: there's an sklearn tool for computing the MSE)
MSE = []
weights = []
for i in range(-10,11):
    new_weight = weight + i
    weights.append(new_weight)
    y_new_pred = new_weight * X + bias
    mse = mean_squared_error(y, y_new_pred)
    MSE.append(mse)
    
print(MSE)
print(weights)
plt.scatter(weights, MSE  , color="black")
plt.xlabel('weights')
plt.ylabel('MSE')
plt.show()
    


# %%
weight_adjustment = range(-10,10)
# %%
# Practical - classification dataset
# Load in the breast cancer dataset from sklearn
# Find a classification model in sklearn
# Initialise the model
# Fit the model 
# Get the score on the training data
# Print a prediction made by the fitted model

from sklearn import datasets
data = datasets.load_breast_cancer()
print(data.keys())
print(data.DESCR)

import pandas as pd
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.head()
df.info()
# %%
# Store the feature data
X = data.data
# store the target data
y = data.target
# split the data using Scikit-Learn's train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
# %%
from sklearn.neighbors import KNeighborsClassifier
logreg = KNeighborsClassifier(n_neighbors=6)
logreg.fit(X_train, y_train)
logreg.score(X_test, y_test)
# %%
