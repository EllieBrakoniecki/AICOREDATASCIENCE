#%%
from sklearn import datasets
 
X, y = datasets.load_boston(return_X_y=True)
print(X)
#%%
print(y)
#%%
print(X.shape)
#%%
print(y.shape)
# %%
# California housing data set

from sklearn import datasets

X, y = datasets.fetch_california_housing(return_X_y=True, as_frame=True)
X
# %%
y.shape
# %%
X.shape
# %%
y[:5]
# %%
X[:5]
X.head()
# %%
X.describe()
# %%
