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
alphas = np.linspace(15, 19, 100)
for i in alphas:
    lr = LogisticRegression(C=i, max_iter=1000)
    cv = cross_val_score(lr, X, y, cv=8)
    lr_cv.append(cv.mean())
plt.plot(alphas, lr_cv, label="Logistic Regression")
plt.legend()
plt.show()


#Logistic Regression, Train Test Split, .33 test split
train_scores = []
test_scores = []
for k in range(0, 100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=k)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)
    train_scores.append(lr.score(X_train, y_train))
    test_scores.append(lr.score(X_test, y_test))
state = []
for i in range(0, 100):
    state.append(i)
plt.plot(state, test_scores, label='test')
plt.plot(state, train_scores, label='train')
plt.legend()
plt.show()



