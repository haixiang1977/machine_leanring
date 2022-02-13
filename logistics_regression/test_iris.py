import numpy as np
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split

# load data
iris = datasets.load_iris()
# use 1st 2 features
x = iris.data[:, :2]
y = iris.target

# split training and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# standarlize feature values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

# logistics regression modeling
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(x_train, y_train)

# predict
predicted = logreg.predict_proba(x_test_std)
# score
acc = logreg.score(x_test_std, y_test)
print("acc ", acc)
