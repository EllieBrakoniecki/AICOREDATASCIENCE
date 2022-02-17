#%%
# https://towardsdatascience.com/https-medium-com-chayankathuria-optimization-ordinary-least-squares-gradient-descent-from-scratch-8b48151ba756
import numpy as np
#%%
# x = np.array([1,3,5])
# y = np.array([5,12,18])
x = np.array([1,3,5,7,9,11])
y = np.array([5,12,18,24,31,37])
# %%
# initialise weights and step size

b_new = 0
w1_new = 0
a = 0.001 
# a = 0.04

MSE = np.array([])
# %%
# After initialising we iterate throught the data set multiple times
# and calculate mean square error per iteration and update the weights 
# BATCH GRADIENT DESCENT
num_iterations = 10

for i in range(1, num_iterations+1):
    y_pred = np.array([]) # The predicted target
    error = np.array([])  # The errors per iteration (Yhat - Y)
    error_x = np.array([]) # The (Yhat - Y).X term for the update rule
    
    b = b_new    # step 4: assigning the updated weights 
    w1 = w1_new
    
    for value in x: 
        y_pred = np.append(y_pred, b + (w1 * value)) # iterating row by row to calculate Yhat
        
    error = np.append(error, y_pred - y) 
    error_x = np.append(error_x, error * x) # value of the gradient wrt w1
    MSE_val = (error**2).mean()
    MSE = np.append(MSE, MSE_val)
    
    b_new = b - a * np.sum(error)
    w1_new = w1 - a * np.sum(error_x)
    
    
print("MSE:",MSE)     
print("b:",b_new)
print("w1:",w1_new)

# check the predicted target variable, Ŷ and the Error:
print("y_pred", y_pred)
print("error:", error)
print("actual y: ", y)

import matplotlib.pyplot as plt
plt.plot(MSE, 'b-o')
plt.title("Mean square error per iteration")
plt.xlabel("Iterations")
plt.ylabel("MSE value")
plt.show()

   
# %%
# STOCHASTIC GRADIENT DESCENT
x = np.array([1,3,5])
y = np.array([5,12,18])

# initialise weights and step size
b_new = 0
w1_new = 0
a = 0.04
num_iterations = 15

for i in range(1, num_iterations+1):
    y_pred = 0
    error = 0
    error_x = 0
    
    for i, number in enumerate(x): # stochastic so the updates are done every iteration
        b = b_new    # step 4: assigning the updated weights 
        w1 = w1_new
        y_pred = b + (w1 * number)     
        error = y_pred - y[i]
        error_x = error * number # value of the gradient wrt w1
        b_new = b - a * error
        w1_new = w1 - a * error_x
        mse = (y_pred - y[i])**2
        print(mse)

b_estimate = b_new
w1_estimate = w1_new

y_prediction = np.array(b_estimate + w1_estimate * x)
error = y_prediction - y
MSE = (error **2).mean()
        
print('estimate of b:',b_estimate)
print('estimate of w1:',w1_estimate)

# # check the predicted target variable, Ŷ and the Error:
print("y_pred", y_prediction)
print("actual y: ", y)
print("error:",error)
print("MSE value:", MSE)

import matplotlib.pyplot as plt
plt.plot(MSE, 'b-o')
plt.title("Mean square error per iteration")
plt.xlabel("Iterations")
plt.ylabel("MSE value")
plt.show()


# %%
# MINI BATCH GRADIENT DESCENT
import numpy as np
x = np.array([1,3,5,7,9,11])
y = np.array([5,12,18,24,31,37])
b_new = 0
w1_new = 0
a = 0.001
MSE = np.array([])
import math
batch_size = 2
number_of_batches = math.ceil(len(x)/batch_size)
print(number_of_batches)

num_iterations = 15
for iteration in range(1, num_iterations+1):
    i = 0    
    for j in range(number_of_batches):
        # print(i)
        print('j:',j)
        y_pred = np.array([]) # The predicted target
        error = np.array([])  # The errors per iteration (Yhat - Y)
        error_x = np.array([]) # The (Yhat - Y).X term for the update rule
    
        b = b_new    # step 4: assigning the updated weights 
        w1 = w1_new
        k = (j*batch_size)+batch_size
        for value in x[i:k]:
            print('x[i:k]',x[i:k]) 
            y_pred = np.append(y_pred, b + (w1 * value)) # iterating row by row to calculate Yhat    
            print('y_pred', y_pred)    
        error = np.append(error, y_pred - y[i:k]) 
        print("error",error)
        error_x = np.append(error_x, error * x[i:k]) # value of the gradient wrt w1
        MSE_val = (error**2).mean()
        MSE = np.append(MSE, MSE_val)    
        b_new = b - a * np.mean(error)
        w1_new = w1 - a * np.mean(error_x)
        i +=batch_size
    
    
print("MSE:",MSE)     
b_estimate = b_new
w1_estimate = w1_new

y_prediction = np.array(b_estimate + w1_estimate * x)
error = y_prediction - y
MSE = (error **2).mean()
        
print('estimate of b:',b_estimate)
print('estimate of w1:',w1_estimate)

# # check the predicted target variable, Ŷ and the Error:
print("y_pred", y_prediction)
print("actual y: ", y)
print("error:",error)
print("MSE value:", MSE)



# %%
