print('#########################################################################')
print('################ Probability Estimates with go.Contour  #################')
print('#########################################################################')

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

mesh_size = .02
margin = 0.25

# Load and split data
cancer = pd.read_csv('././data/cancer.csv')
cancer.replace({"diagnosis":{'B':0,'M':1}}, inplace=True)
cancer = cancer.drop(columns=['id'])
print(cancer.head())
X = np.array(cancer.drop(['diagnosis'], 1))
y = np.array(cancer['diagnosis'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y.astype(str), test_size=0.3, random_state=0)

# Create a mesh grid on which we will run our model
x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(xrange, yrange)

# Create classifier, run predictions on grid
clf = KNeighborsClassifier(15, weights='uniform')
clf.fit(X_train, y_train)
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)


# Plot the figure
fig = go.Figure(data=[
    go.Contour(
        x=xrange,
        y=yrange,
        z=Z,
        colorscale='RdBu'
    )
])
fig.show()
