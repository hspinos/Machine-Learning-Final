# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#Load data
data = np.loadtxt('leaf.csv', delimiter=',')
X = data[: , 2:15]
y = data[: , 0]

#Model Using Train Test Split

C_vals1 = np.linspace(.1, 100, 101)
scores1 = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=1)
for c in C_vals1:
    lr = LogisticRegression(C=c, solver='newton-cg', max_iter=100, random_state=1)
    lr.fit(X_train, y_train)
    scores1.append(lr.score(X_test, y_test))

plt.plot(C_vals1, scores1)

#Model Using 8-Fold Cross Validation

C_vals2 = np.linspace(.1, 100, 101)
scores2 = []
for c in C_vals2:
    lr = LogisticRegression(C=c, solver='newton-cg', max_iter=100, random_state=1)
    scores2.append(cross_val_score(lr, X, y, cv=8).mean());

plt.figure()
plt.plot(C_vals2, scores2)