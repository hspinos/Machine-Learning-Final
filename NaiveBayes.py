# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

#Load data
data = np.loadtxt('leaf2.csv', delimiter=',')
X = data[: , 2:15]
y = data[: , 0]

#Model Using Train Test Split

varSmooths = np.linspace(0, .001, 101)

#for r in range(1, 11):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=1)
train_scores = []
test_scores = []
for v in varSmooths:
    GNB = GaussianNB(var_smoothing=v)
    GNB.fit(X_train, y_train)
    train_scores.append(GNB.score(X_train, y_train))
    test_scores.append(GNB.score(X_test, y_test))
    
plt.plot(varSmooths, train_scores, label="Train")
plt.plot(varSmooths, test_scores, label="Test")
plt.title("GNB Train Test Split")
plt.legend()
    

#Model Using 8-Fold Cross Validation

varSmooths = np.linspace(0, .001, 101)
cv_scores = []
for v in varSmooths:
    GNB2 = GaussianNB(var_smoothing=v)
    cv = cross_val_score(GNB2, X, y, cv=8)
    cv_scores.append(cv.mean())
    
plt.figure()
plt.plot(varSmooths, cv_scores, label="Scores")
plt.title("GNB 8-Fold CV")
plt.legend()

