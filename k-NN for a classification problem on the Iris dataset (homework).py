# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 20:17:21 2020

@author: Ørjan
"""
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, model_selection
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Load in Iris dataset
#Splitting the datasets into training, validation and testing
iris = load_iris()
X, y = iris.data[:, 2], iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, random_state=0)

N_train, N_val, N_test = len(X_train), len(X_val), len(X_test)
#print(N_train, N_val, N_test)

cmap_light = ListedColormap(['yellow', 'cyan', 'pink'])
cmap_bold = ListedColormap(['red', 'darkcyan', 'darkblue'])

#Plotting the trained datasets
ax = plt.gca()
ax.scatter(X_test, y_test, c=y_test, cmap=cmap_light, edgecolor='k', s=20, zorder=2)
ax.scatter(X_val, y_val, c=y_val, cmap=cmap_light, edgecolor='k', s=20, zorder=2)
ax.scatter(X_train, y_train, c=y_train, cmap=cmap_bold, edgecolor='k', s=20, zorder=2)

plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.tight_layout()
plt.show()

#formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
#plt.scatter(iris.data[:, 0], iris.data[:, 1], c=y)
#plt.scatter(X_test, y_test, c=y_test)
#plt.scatter(X_train, y_train, c=y_train)
#plt.colorbar(ticks=[0, 1, 2], format=formatter)

#Computing model accuracy on training dataset for different k-value
scores = {}
scores_list = []
for k in range(1, 31):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train.reshape(-1,1), y_train)
    y_pred = knn.predict(X_test.reshape(-1,1))
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred)) 
plt.plot(range(1, 31), scores_list)
plt.xlabel('Value of K for kNN')
plt.ylabel('Testing Accuracy')
plt.tight_layout()

#Giving k the value for highest testing accuracy
knn = neighbors.KNeighborsClassifier(n_neighbors=29)
knn.fit(X.reshape(-1,1), y)


'''
(Validation dataset is too small to be used?)

#Computing model accuracy on validation dataset for different k-value
scores = {}
scores_list = []
for k in range(1, 31):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_val.reshape(-1,1), y_val)
    y_pred = knn.predict(X_test.reshape(-1,1))
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))    
plt.plot(range(1, 31), scores_list)
'''