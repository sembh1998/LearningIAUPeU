"""
Clasificador de iris
@sembh1998
"""

#lybraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling

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

X = np.array(iris.drop(['Species'], 1))
y = np.array(iris['Species'])


kmeans = KMeans(n_clusters= 3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()

print('==================')