# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Preprocessing Data
data = np.loadtxt('leaf.csv', delimiter=',')
X = data[:, 2:15]
y = data[:, 0]

# Logistic Regression, K-Fold 8 Cross Validation
lr_cv = []
alphas = np.linspace(15,17,20)
for i in alphas:
    lr = LogisticRegression(C=i, max_iter=1000)
    cv = cross_val_score(lr, X, y, cv=8)
    lr_cv.append(cv.mean())
plt.plot(alphas, lr_cv, label="Logistic Regression")
plt.legend()
plt.show()
