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


