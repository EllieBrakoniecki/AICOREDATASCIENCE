#%%

# https://matplotlib.org/stable/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()
#You may be wondering why the x-axis ranges from 0-3 and the y-axis from 1-4.
# If you provide a single list or array to plot, matplotlib assumes it is a sequence of y values, and automatically generates the x values for you. Since python ranges start with 0, the default x vector has the same length as y but starts with 0. Hence the x data are [0, 1, 2, 3].
# %%
# plot is a versatile function, and will take an arbitrary number of arguments. For example, to plot x versus y, you can write:
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
# %%
