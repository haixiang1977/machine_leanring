import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# generate source data
x = 2 * np.random.rand(100, 1)
# y = 4 + 3x
# thelta_0 = 4, thelta_1 = 3
y = 4 + 3 * x + np.random.randn(100, 1)

print("plot *x, y)")
plt.scatter(x, y)
plt.show()

# add x0 = 1 to each instance
# c_ numpy link two arrays
xb = np.c_[np.ones((100, 1)), x]
# numpy array transpose
# Normal Equation:
# theta = inv(X^T * X) * X^T * y
xb_transpose = np.transpose(xb)
# numpy dot product
thelta = inv(xb_transpose.dot(xb))
thelta = thelta.dot(xb_transpose)
thelta = thelta.dot(y)
print("thelta: ", thelta)
