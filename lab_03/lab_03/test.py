import numpy as np

x = np.arange(12).reshape(3, 4)
y = np.array([2, 1, 3])
z = x
print(x)
print(y)
print(np.average(x, axis=1))
print(np.average(x, axis=0))
for i in range(x.shape[0]):
    x[int(i)] -= y[int(i)]
x.sub(y)
print(x)
print(z)
