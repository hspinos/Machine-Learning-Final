# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

#Load data
data = np.loadtxt('leaf.csv', delimiter=',')
X = data[: , 2:15]
y = data[: , 0]

#Model Using Train Test Split

varSmooths = np.linspace(0, .001, 101)

for r in range(1, 11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=r)
    scores = []
    for v in varSmooths:
        GNB = GaussianNB(var_smoothing=v)
        GNB.fit(X_train, y_train)
        scores.append(GNB.score(X_test, y_test))
    plt.plot(varSmooths, scores)
    



#Model Using 10-Fold Cross Validation