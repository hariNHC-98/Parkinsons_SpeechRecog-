import pandas as pd 
import numpy as np 
import os, sys
#url = 'https://github.com/hariNHC-98/Parkinsons_SpeechRecog-/blob/master/parkinson_10.data.numbers'
#data = pd.read_csv(url)
#data = pd.read_csv('parkinsons.data')

# Training data
from google.colab import files
uploaded = files.upload()

import io
data = pd.read_csv(io.BytesIO(uploaded['parkinson10.data']))
# Dataset is now stored in a Pandas Dataframe

# Train Data Formatting
predictors = data.drop(['name'], axis = 1)
predictors = predictors.drop(['status'], axis = 1).as_matrix()
target = data['status']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((-1, 1))
X = scaler.fit_transform(predictors)
Y = target

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .00, random_state = 7)

# Testing data 
from google.colab import files
uploaded = files.upload()

import io
data2 = pd.read_csv(io.BytesIO(uploaded['yes_park.data']))
# Dataset is now stored in a Pandas Dataframe

#Test data formatting
predictor2 = data.drop(['name'], axis = 1)
predictor2 = predictor2.drop(['status'], axis = 1).as_matrix()
target = data['status']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((-1, 1))
X_test= scaler.fit_transform(predictors)
Y_test = target



# Logistic Regression
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# fit a logistic regression model to the data
model = LogisticRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
# summarize the fit of the model
print("Logistic Regression: ")
print(metrics.accuracy_score(Y_test, y_pred))
print(metrics.classification_report(Y_test, y_pred))
print(metrics.confusion_matrix(Y_test, y_pred))

# Gaussian Naive Bayes
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, Y_train)
# make predictions
y_pred = model.predict(X_test)
# summarize the fit of the model
print("Gaussian Naive Bayes:")
print(metrics.accuracy_score(Y_test, y_pred))
print(metrics.classification_report(Y_test, y_pred))
print(metrics.confusion_matrix(Y_test, y_pred))


#K-Nearest Neighbor (SUPPOSEDLY BEST)
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
# make predictions
y_pred = model.predict(X_test)
# summarize the fit of the model
print("k-Nearest Neighbor: ")
print(metrics.accuracy_score(Y_test, y_pred))
print(metrics.classification_report(Y_test, y_pred))
print(metrics.confusion_matrix(Y_test, y_pred))

# Support Vector Machine
from sklearn import datasets
from sklearn import metrics
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, Y_train)

# make predictions
y_pred = model.predict(X_test)
# summarize the fit of the model
print("Support Vector Machine: ")
print(metrics.accuracy_score(Y_test, y_pred))
print(metrics.classification_report(Y_test, y_pred))
print(metrics.confusion_matrix(Y_test, y_pred))

# Classification and Regression Trees
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
# make predictions
y_pred = model.predict(X_test)
# summarize the fit of the model
print("Classification and Regression Trees")
print(metrics.accuracy_score(Y_test, y_pred))
print(metrics.classification_report(Y_test, y_pred))
print(metrics.confusion_matrix(Y_test, y_pred))
