# Process database and Train classifiers basing on extracted features
import numpy as np
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn import tree
import sklearn
import os
import pickle
import random
import time

# Loading Dataset
Label = np.load('Label.npy')
index_test = np.load('index_test.npy')
index_train = np.load('index_train.npy')
path = "DATA"
Files_name = os.listdir(path)

Train = [];Dev = [];Test = [];Y_train = [];Y_test = []
Proporation = 1 # Proportion of (Train +Validation)
# Train set
for i in index_train:
    file = Files_name[i]
    data = np.load(path+'/'+file)
    if data.size == 140*28:
        data = data.reshape(1, 140*28)[0]
        Train.append(data)
        Y_train.append(Label[i])
    else:
        pass
# Test set
for i in index_test:
    file = Files_name[i]
    data = np.load(path+'/'+file)
    if data.size == 140*28:
        data = data.reshape(1, 140*28)[0]
        Test.append(data)
        Y_test.append(Label[i])
    else:
        pass

# Load Models and Parameters
random.seed(100) # 这里设了seed结果还是会在报告值附近波动
DT = tree.DecisionTreeClassifier(min_weight_fraction_leaf = 0.15) 
SVM = SVC(C=1e-6, kernel='linear', gamma=1e-6)
LR = sklearn.linear_model.LogisticRegression
LM = LR(penalty = 'l1')

# Fit Models
DT.fit(Train,Y_train)
LM.fit(Train,Y_train)
SVM.fit(Train,Y_train) # SVM训练所需时间较长

predict = LM.predict(Test)
precision = sklearn.metrics.precision_score(Y_test, predict)
recall = sklearn.metrics.recall_score(Y_test, predict)
accuracy = sklearn.metrics.accuracy_score(Y_test, predict)
print('=== Logistic Regression ===')
print('precision: %.2f%%\nrecall: %.2f%%' % (100 * precision, 100 * recall))
print('accuracy: %.2f%%' % (100 * accuracy))
predict = DT.predict(Test)
precision = sklearn.metrics.precision_score(Y_test, predict)
recall = sklearn.metrics.recall_score(Y_test, predict)
accuracy = sklearn.metrics.accuracy_score(Y_test, predict)
print('=== Decision Tree ===')
print('precision: %.2f%%\nrecall: %.2f%%' % (100 * precision, 100 * recall))
print('accuracy: %.2f%%' % (100 * accuracy))
predict = SVM.predict(Test)
precision = sklearn.metrics.precision_score(Y_test, predict)
recall = sklearn.metrics.recall_score(Y_test, predict)
accuracy = sklearn.metrics.accuracy_score(Y_test, predict)
print('=== Support Vector Machine ===')
print('precision: %.2f%%\nrecall: %.2f%%' % (100 * precision, 100 * recall))
print('accuracy: %.2f%%' % (100 * accuracy))

