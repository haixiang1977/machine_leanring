# use sklearn polynomial feature not linear regression
import numpy as np
import matplotlib.pyplot as plt

# generate source data
x = np.random.uniform(-3, 3, size = 100)
# transform to [100 x 1] array
x_reshape=x.reshape(-1, 1)
y = 0.5 + x**2 + x + 2 + np.random.normal(0, 1, size = 100)

# start to use polynomial feature and pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# try degree = 2 polynomial
print("degree = 2")
poly_reg2 = Pipeline([
    ('poly', PolynomialFeatures(degree = 2)),
    ('std_scale', StandardScaler()),
    ('lin_reg', LinearRegression())
])
poly_reg2.fit(x_reshape, y)
y_predict2 = poly_reg2.predict(x_reshape)
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')
plt.show()

print("degree = 10")
poly_reg10 = Pipeline([
    ('poly', PolynomialFeatures(degree = 10)),
    ('std_scale', StandardScaler()),
    ('lin_reg', LinearRegression())
])
poly_reg10.fit(x_reshape, y)
y_predict10 = poly_reg10.predict(x_reshape)
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict10[np.argsort(x)], color='r')
plt.show()

print("degree = 100")
poly_reg100 = Pipeline([
    ('poly', PolynomialFeatures(degree = 100)),
    ('std_scale', StandardScaler()),
    ('lin_reg', LinearRegression())
])
poly_reg100.fit(x_reshape, y)
y_predict100 = poly_reg100.predict(x_reshape)
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict100[np.argsort(x)], color='r')
plt.show()

