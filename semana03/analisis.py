import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# loading the data from csv file to a Pandas DataFrame
calories = pd.read_csv('./data/calories.csv')

# print the first 5 rows of the dataframe
print(calories.head())

exercise_data = pd.read_csv('./data/exercise.csv')

print(exercise_data.head())

calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

print(calories_data.head())

# checking the number of rows and columns
print("Number of rows and columns: ", calories_data.shape)

# getting some informations about the data
print("Calories Info:\n")
print(calories_data.info())

# checking for missing values
print("is there missing values?\n",calories_data.isnull().sum())

# get some statistical measures about the data
print("Statistical measures:\n", calories_data.describe())

#converting the text data to numerical values
calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)

X = np.array(calories_data.drop(['Calories'], 1))
y = np.array(calories_data['Calories'])

#Separating the data of the training and the test of the algorithms
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print('\n Son {} datos de entrenamiento y {} datos de prueba'.format(X_train.shape[0], X_test.shape[0]))


#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
print('\nLogistic Regression:{}'.format(logreg.score(X_train, y_train)))


#Support Vector Machine
# svc = SVC()
# svc.fit(X_train, y_train)
# Y_pred = svc.predict(X_test)
# print('\nSupport Vector Machine:{}'.format(svc.score(X_train, y_train)))

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


"""
Clasificador de iris
@sembh1998
"""

#data
iris = pd.read_csv('./data/Iris.csv')

#delete the rest of columns
iris = iris.drop('Id', axis=1)
iris = iris.drop('SepalLengthCm', axis=1)
iris = iris.drop('SepalWidthCm', axis=1)
#iris = iris.drop('Species', axis=1)
#print(iris.head())
from sklearn.decomposition import PCA
# USE THE ELKAN ALGORITHM
# The “elkan” variation is more efficient on data with well-defined clusters, by using the triangle inequality
from sklearn.cluster import KMeans

print("==========IRIS================")

X = np.array(iris.drop(['Species'], 1))
y = np.array(iris['Species'])

#Separating the data of the training and the test of the algorithms
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print('\n Son {} datos de entrenamiento y {} datos de prueba'.format(X_train.shape[0], X_test.shape[0]))


kmeans = KMeans(n_clusters= 3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()

print('==================')

#K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
print('\nK-Nearest Neighbors:{}'.format(knn.score(X_train, y_train)))



print("================================================")
print("================================================")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

h = 0.02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]

for weights in ["uniform", "distance"]:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=iris.target_names[y],
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(
        "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
    )
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])

plt.show()