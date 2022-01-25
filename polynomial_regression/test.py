# sudo apt-get install python3-sklearn python3-sklearn-lib 
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# check sklearn install
print("sklearn version ", sk.__version__)

# generate data source
print("generate data source")
x = np.random.uniform(-3, 3, 100)
y = 0.5 + x**2 + x + 2 + np.random.normal(0, 1, size=100)
plt.scatter(x, y)
plt.show()

# convert x to 100 x 1 matrix
x_reshape = x.reshape(-1, 1)

# now try linear regression
print("try linear regression y = thelta_0 + thelta_1*x")
lin_reg = LinearRegression()
lin_reg.fit(x_reshape, y)
y_predict = lin_reg.predict(x_reshape)
# show linear regression parameter
# coef is thelta_1
print("coef ", lin_reg.coef_)
# intercept is thelta_0 (the value of y when x = 0)
print("intercept ", lin_reg.intercept_)
plt.scatter(x, y)
plt.plot(x, y_predict, color='r')
plt.show()

# now try polynomial regression
# add one more feature varilable x**2
print("try polynomial regression y = thelta_0 + thelta_1*x + thelta_2*x**2")
x2 = np.hstack([x_reshape, x_reshape**2])
lin_reg2 = LinearRegression()
lin_reg2.fit(x2, y)
y_predict2 = lin_reg2.predict(x2)
plt.scatter(x, y)
# show polynomial parameter
# coef is thelta_1 and thelta_2
print("coef ", lin_reg2.coef_)
# intercept is thelta_0 (the value of y when x = 0)
print("intercept ", lin_reg2.intercept_)
# sort just sorted value
# argsort return the index of sorted value
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')
plt.show()
