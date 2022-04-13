"""
Clasificador de iris
@sembh1998
"""

#lybraries
import numpy as np
import pandas as pd

#data
iris = pd.read_csv('./data/Iris.csv')

#delete the first column
iris = iris.drop('Id', axis=1)
#print(iris.head())

#analysing the data
print('Info:')
print(iris.info())

print('\nDescriptions:')
print(iris.describe())

print('\nSpecies:')
print(iris.groupby('Species').size())

#visualizing the data

import matplotlib.pyplot as plt

#Sépalo graphic - Longitud vs Ancho
fig = iris[iris.Species == 'Iris-setosa'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='blue', label='Setosa')
iris[iris.Species == 'Iris-versicolor'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='red', label='Versicolor', ax=fig)
iris[iris.Species == 'Iris-virginica'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='green', label='Virginica', ax=fig)

fig.set_xlabel('Sépalo - Longitud')
fig.set_ylabel('Sépalo - Ancho')
fig.set_title('Sépalo - Longitud vs Ancho')
plt.show()

#Pétalo graphic - Longitud vs Ancho
fig2 = iris[iris.Species == 'Iris-setosa'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='blue', label='Setosa')
iris[iris.Species == 'Iris-versicolor'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='red', label='Versicolor', ax=fig2)
iris[iris.Species == 'Iris-virginica'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='green', label='Virginica', ax=fig2)

fig2.set_xlabel('Pétalo - Longitud')
fig2.set_ylabel('Pétalo - Ancho')
fig2.set_title('Pétalo - Longitud vs Ancho')
plt.show()

#Aplying the machine learning algorithms

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

pca = PCA(2)

#delete the last column
iris2 = iris.drop('Species', axis=1)

df = pca.fit_transform(iris2)
df.shape

kmeans = KMeans(n_clusters= 3)

#predict the labels of clusters.
label = kmeans.fit_predict(df)
 
#Getting unique labels
u_labels = np.unique(label)
 
#plotting the results:
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()
plt.show()

print('==================')
#Models with the data
X = np.array(iris.drop(['Species'], 1))
y = np.array(iris['Species'])

#Separating the data of the training and the test of the algorithms
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('\n Son {} datos de entrenamiento y {} datos de prueba'.format(X_train.shape[0], X_test.shape[0]))

#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
print('\nLogistic Regression:{}'.format(logreg.score(X_train, y_train)))


#Support Vector Machine
svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
print('\nSupport Vector Machine:{}'.format(svc.score(X_train, y_train)))

#K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
print('\nK-Nearest Neighbors:{}'.format(knn.score(X_train, y_train)))

#Decision Tree
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
Y_pred = dtree.predict(X_test)
print('\nDecision Tree:{}'.format(dtree.score(X_train, y_train)))


