# https://towardsdatascience.com/gradient-descent-the-learning-rate-and-the-importance-of-feature-scaling-6c0b416596e1


# linear regression model y = b + wx + e
# in our model use feature(x) to predict value of a label y
# 3 elements in our model: 
# parameter b, bias or intercept, tells us the expected value of y when x is 0
# parameter w, the weight or slope which tells us how much y increases, on average, if we increase x by one unit
# e which is there to account for inherent noise


# Generate synthetic data: Let b = 1 and w = 2
# Generate feature x: use numpy rand to randomly generate 100(N) points between 0 and 1
# Then, we plug our feature (x) and our parameters b and w into our equation to compute our labels (y). But we need to add some Gaussian noise (epsilon) as well; otherwise, our synthetic dataset would be a perfectly straight line.

# We can generate noise using Numpy’s randn method, which draws samples from a normal distribution (of mean 0 and variance 1), and then multiplying it by a factor to adjust for the level of noise. 
# Since I don’t want to add so much noise, I picked 0.1 as my factor.

#%%
import numpy as np

true_b = 1
true_w = 2
N = 100

# Data Generation
np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = (.1 * np.random.randn(N, 1))
y = true_b + true_w * x + epsilon
# %%
# Train-Validation-Test Split

# Shuffles the indices
idx = np.arange(N)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:int(N*.8)]
# Uses the remaining indices for validation
val_idx = idx[int(N*.8):]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]


import matplotlib.pyplot as plt
plt.plot(x_train,y_train,'bo')
plt.xlabel("x_train")
plt.ylabel("y_train")
plt.show()

plt.plot(x_val,y_val,'ro')
plt.xlabel("x_val")
plt.ylabel("y_val")
plt.show()


# #%%
# Random Initialization
# In our example, we already know the true values of the parameters, but this will obviously never happen in real life: if we knew the true values, why even bother to train a model to find them?!
# OK, given that we’ll never know the true values of the parameters, we need to set initial values for them. How do we choose them? It turns out; a random guess is as good as any other.
# So, we can randomly initialize the parameters/weights (we have only two, b and w).
# %%
# Initializes parameters "b" and "w" randomly
np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(1)

# Our randomly initialized parameters are: b = 0.49 and w = -0.13. Are these parameters any good?
import numpy as np
import matplotlib.pyplot as plt

plt.plot(x_train,y_train,'bo')
plt.xlabel("x_train")
plt.ylabel("y_train")
plt.plot(x_train, w*x_train + b) 
plt.show()

#Obviously not… but, exactly how bad are they? That's what the loss is for. Our goal will be to minimize it.
# %%
#Loss Surface
# After choosing a random starting point for our parameters, we use them to make predictions, compute the corresponding errors, and aggregate these errors into a loss. Since this is a linear regression, we’re using Mean Squared Error (MSE) as our loss. The code below performs these steps:
# Computes our model's predicted output - forward pass
yhat = b + w * x_train

# Computing the loss
# We are using ALL data points, so this is BATCH gradient
# descent. How wrong is our model? That's the error!
error = (yhat - y_train)

# It is a regression, so it computes mean squared error (MSE)
loss = (error ** 2).mean()

print(loss)
# %%
# We have just computed the loss (2.74) corresponding to our randomly initialized parameters (b = 0.49 and w = -0.13). 
# Now, what if we did the same for ALL possible values of b and w? 
# Well, not all possible values, but all combinations of evenly spaced values in a given range?

# We could vary b between -2 and 4, while varying w between -1 and 5, for instance, each range containing 101 evenly spaced points. 
# If we compute the losses corresponding to each different combination of the parameters b and w inside these ranges, 
# the result would be a grid of losses, a matrix of shape (101, 101).

# These losses are our loss surface, which can be visualized in a 3D plot,
# where the vertical axis (z) represents the loss values. If we connect the combinations 
# of b and w that yield the same loss value, we’ll get an ellipse. Then, we can draw this
# ellipse in the original b x w plane (in blue, for a loss value of 3). 
# This is, in a nutshell, what a contour plot does. From now on, we’ll always use the contour plot, instead of the corresponding 3D version.

# This is one of the nice things about tackling a simple problem like a linear
# regression with a single feature: we have only two parameters, and thus we can compute and visualize the loss surface.

# Cross-Sections
# Another nice thing is that we can cut a cross-section in the loss surface to check what the loss looks like if the other parameter were held constant.



# Updating Parameters
# Finally, we use the gradients to update the parameters. 
# Since we are trying to minimize our losses, 
# we reverse the sign of the gradient for the update.
# There is still another (hyper-)parameter to consider: 
# the learning rate, denoted by the Greek letter eta (that looks like the letter n), 
# which is the multiplicative factor that we need to apply to the gradient for the parameter update.

# We can also interpret this a bit differently: each parameter is going to have its
# value updated by a constant value eta (the learning rate), but this constant is going to be weighted by how much that parameter contributes to minimizing the loss (its gradient).
# Honestly, I believe this way of thinking about the parameter update makes more sense: 
# first, you decide on a learning rate that specifies your step size, while the gradients 
# tell you the relative impact (on the loss) of taking a step for each parameter. 
# Then you take a given number of steps that’s proportional to that relative impact: 
# more impact, more steps.


# Too big, for a learning rate, is a relative concept: it depends on how steep the curve is or, 
# in other words, it depends on how big the gradient is.
# We do have many curves, many gradients: one for each parameter. 
# But we only have one single learning rate to choose (sorry, that’s the way it is!).
# It means that the size of the learning rate is limited by the steepest curve. 
# All other curves must follow suit, meaning, they’d be using a sub-optimal learning rate, 
# given their shapes.
# The reasonable conclusion is: it is best if all the curves are equally steep, so the learning rate is closer to optimal for all of them!


# How do we achieve equally steep curves? I’m on it! 
# First, let’s take a look at a slightly modified example, which I am calling the “bad” dataset:


# Scaling / Standardizing / Normalizing
# Different how? There is this beautiful thing called the StandardScaler, which transforms a feature in such a way that it ends up with zero mean and unit standard deviation.
# How does it achieve that? First, it computes the mean and the standard deviation of a given feature (x) using the training set (N points):


see link above for further details