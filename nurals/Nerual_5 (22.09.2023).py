import numpy as np

x = [1, 2, 3]
y = [3, 4, 5]
z = []

for k in range(len(x)):
    z.append(x[k]*y[k])

print(z)

x1 = np.array([1, 2, 3])
y1 = np.array([3, 4, 5])

z1 = x1 * y1
print(z1)

a1D = np.array([1, 2, 3, 4])
a2D = np.array([[1, 2], [3, 2]])
a3D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(a1D,a2D,a3D)