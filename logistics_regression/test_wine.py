# SciPy is a free and open-source Python library used for scientific computing and technical computing
import scipy as sp
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# x is wine property
x=np.loadtxt("wine.data", delimiter = ",", usecols = [1,2,3,4,5,6,7,8,9,10,11,12,13])
# y is wine label
y=np.loadtxt("wine.data", delimiter = ",", usecols = [0])
# check sample
# print(x)
# print(y)

# seperate train samples for 80% and test samples for 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# logistics regression
model = LogisticRegression()
model.fit(x_train, y_train)
# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False)
print(model)

# make predictions
# expected output
expected = y_test
# predicted from test samples with modeling
predicted = model.predict(x_test)

# check difference between predict and expected
#             precision    recall  f1-score   support
#
#        1.0       1.00      1.00      1.00         8
#        2.0       1.00      1.00      1.00        17
#        3.0       1.00      1.00      1.00        11
#
# avg / total       1.00      1.00      1.00        36
print(metrics.classification_report(expected, predicted))
# [[10  1  0]
#  [ 0 15  0]
#  [ 0  0 10]]
print(metrics.confusion_matrix(expected, predicted))

