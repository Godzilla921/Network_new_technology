import numpy as np

x = np.arange(12).reshape(3, 4)
y = np.arange(12).reshape(4, 3)

print(x)
print(y)
print(x[1, :])
print(y[:, 2])
print(np.sum(x[1, :]))
print(np.sum(y[:, 2]))
print(np.sum(x[1, :] * y[:, 2]))
