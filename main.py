from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
iris_data = iris.data
iris_target = iris.target

MLPclf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,10, 5), random_state=1)
MLPclf.fit(iris_data,iris_target)
iris_predict1 = MLPclf.predict(iris.data)
cnfmat1 = metrics.confusion_matrix(iris_target,iris_predict1)
print("Confusion matrix for MLP classifier:\n")
print(cnfmat1)
print('\n')

SVMclf = svm.SVC(kernel='linear', C=0.01)
SVMclf.fit(iris_data,iris_target)
iris_predict2 = SVMclf.predict(iris.data)
cnfmat2 = metrics.confusion_matrix(iris_target,iris_predict2)
print("Confusion matrix for SVM classifier:\n")
print(cnfmat2)
print('\n')

KNNclf = KNeighborsClassifier(n_neighbors=3)
KNNclf.fit(iris_data,iris_target)
iris_predict3 = KNNclf.predict(iris.data)
cnfmat3 = metrics.confusion_matrix(iris_target,iris_predict3)
print("Confusion matrix for KNN classifier:\n")
print(cnfmat3)
print('\n')

NBclf = GaussianNB()
NBclf.fit(iris_data,iris_target)
iris_predict4 = NBclf.predict(iris.data)
cnfmat4 = metrics.confusion_matrix(iris_target,iris_predict4)
print("Confusion matrix for Naive Bayes classifier:\n")
print(cnfmat4)
print('\n')