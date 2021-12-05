import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# Preprocessing Data
data = np.loadtxt('leaf2.csv', delimiter=',')
X = data[:, 2:15]
y = data[:, 0]

skf = StratifiedKFold()

# Logistic Regression, SKF Cross Validation
lr_cv = []
skf.get_n_splits(X,y)
c_vals = np.linspace(.1, 5, 100)
for i in c_vals:
    lr = LogisticRegression(C=i, max_iter=1000)
    cv = cross_val_score(lr, X, y, cv=skf)
    lr_cv.append(cv.mean())
plt.plot(c_vals, lr_cv, label="Logistic Regression")
plt.title("Logistic Regression SKF Cross Validation")
plt.legend()
plt.show()


#Logistic Regression, Train Test Split, .33 test split
train_scores = []
test_scores = []
c_vals2 = np.linspace(50, 60, 100)
for k in c_vals2:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    lr = LogisticRegression(C = k, max_iter=2000)
    lr.fit(X_train, y_train)
    train_scores.append(lr.score(X_train, y_train))
    test_scores.append(lr.score(X_test, y_test))
plt.plot(c_vals2, test_scores, label='test')
plt.plot(c_vals2, train_scores, label='train')
plt.title("Logistic Regression Train Test Split")
plt.legend()
plt.show()

