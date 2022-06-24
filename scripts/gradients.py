import numpy as np

from mlutils.losses.regression import mse
from mlutils.algos import gradients

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

cost = mse(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w: {cost}')

grad = gradients.mse(X_train, y_train, w_init, b_init)
print(grad)
