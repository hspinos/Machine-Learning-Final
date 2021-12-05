import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = np.loadtxt('leaf.csv', delimiter=',')
X = data[:, 2:15]
y = data[:, 0]

scaler = StandardScaler()
X = scaler.fit_transform(X)

#SVM cross val score
ml_cv = []
c = np.linspace(1, 20, 100)
for i in c:
    svc = SVC(kernel='rbf', C=i)
    svc.fit(X, y)
    cv = cross_val_score(svc, X, y, cv=8)
    ml_cv.append(cv.mean())
plt.plot(c, ml_cv, label='rbf')
plt.legend()
plt.show()

#SVM train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=1)
train_scores = []
test_scores = []
c = np.linspace(1, 20, 100)
for i in c:
    svm = SVC(kernel='rbf', C=i)
    svm.fit(X_train, y_train)
    train_scores.append(svm.score(X_train, y_train))
    test_scores.append(svm.score(X_test, y_test))
plt.plot(c, test_scores, label = 'test')
plt.plot(c, train_scores, label = 'train')
plt.legend()
plt.show()