# Example to recognize handwritten images of digits
print(__doc__)

# For plotting graphs
import matplotlib.pyplot as plt

# Datasets, classifiers and performance metrics from sklearn
from sklearn import datasets, svm, metrics

# The digits dataset in sklearn
digits = datasets.load_digits()

# The digits dataset contains each image as 8 x 8 list in digits.images
# and the digit they represent in digits.target
# zip() function turns each (image, target) pair into a tuple, which is then turned into a list
# of (image, value) tuple
images_and_lables = list(zip(digits.images, digits.target))

for index, (image, label) in enumerate(images_and_lables[:4]):
    plt.subplot(2, 4, index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a support vector classifier
classifier = svm.SVC(gamma=0.001)

# For the first half of digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Predict on second half of digits
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s: \n%s\n" % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix: \n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))

for index, (images, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index+5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()