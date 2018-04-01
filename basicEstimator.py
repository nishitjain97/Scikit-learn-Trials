# A simple support vector classifier
from sklearn import datasets, svm

# This is the dataset
digits = datasets.load_digits()

# This is our classifier
clf = svm.SVC(gamma=0.001, C=100)

# Fitting the classifier to all but the last of the data item
clf.fit(digits.data[:-1], digits.target[:-1])

print(clf.predict(digits.data[-1:]))