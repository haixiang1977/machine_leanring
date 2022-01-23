# sudo apt install python3-pip
# sudo apt-get install python3-numpy
# sudo apt-get install python3-matplotlib
# python3 test.py

import numpy as np
import matplotlib.pyplot as plt

print("test linear regression")

# sample data
X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
Y = [3,4,5,5,2,4,7,8,11,8,10,11,13,13,16,17,16,17,18,20]

# number of sample data
M = 20

# step
Alpha = 0.01

# initial value for thelta_0 and thelta_1
Thelta_0 = 1
Thelta_1 = 1

# thread valule
Thread = 0.001

def predict_fun(thelta_0, thelta_1, x):
    y_predicted = thelta_0 + thelta_1 * x
    return y_predicted

# calcuate error with gradient decent
def loop(thelta_0, thelta_1):
    sum1 = 0
    sum2 = 0
    error = 0
    for i in range(M):
        a = predict_fun(thelta_0, thelta_1, X[i]) - Y[i]
        b = (predict_fun(thelta_0, thelta_1, X[i]) - Y[i]) * X[i]
        error1 = a * a
        sum1 = sum1 + a
        sum2 = sum2 + b
        error = error1 + error
    return sum1, sum2, error

def batch_gradient_descent(thelta_0, thelta_1):
    gradient_1 = (loop(thelta_0, thelta_1)[1]/M)
    while abs(gradient_1) > Thread:
        gradient_0 = (loop(thelta_0, thelta_1)[0]/M)
        gradient_1 = (loop(thelta_0, thelta_1)[1]/M)
        error = (loop(thelta_0, thelta_1)[2]/M)
        thelta_0 = thelta_0 - Alpha * gradient_0
        thelta_1 = thelta_1 - Alpha * gradient_1
    
    return thelta_0, thelta_1, error

Final_thelta_0 = batch_gradient_descent(Thelta_0, Thelta_1)[0]
Final_thelta_1 = batch_gradient_descent(Thelta_0, Thelta_1)[1]
Final_error = batch_gradient_descent(Thelta_0, Thelta_1)[2]

print("Thelta_0 is %f, Thelta_1 is %f, The mean square error is %f" % (Final_thelta_0, Final_thelta_1, Final_error))
# Thelta_0 is 0.985998, Thelta_1 is 0.902374, The mean square error is 1.903032

# draw the figure now
plt.figure(figsize = (6, 4))
# draw sample data
plt.scatter(X, Y, label = 'y')
plt.xlim(0, 21)
plt.ylim(0, 22)
plt.xlabel('x', fontsize = '20')
plt.ylabel('y', fontsize = '20')

x = np.array(X)
y_predict = np.array(Final_thelta_0 + Final_thelta_1 * x)
plt.plot(x, y_predict, color = 'red')
plt.show()
