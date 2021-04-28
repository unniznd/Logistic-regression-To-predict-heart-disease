
!pip install opendatasets

import opendatasets as od
od.download('https://www.kaggle.com/dileep070/heart-disease-prediction-using-logistic-regression')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import math as mt
import seaborn as sns

df = pd.read_csv('/content/heart-disease-prediction-using-logistic-regression/framingham.csv')
df.head()

df.shape

df.info()

df.describe()

df['education'] = df['education'].fillna(1)

df['cigsPerDay'] = df.cigsPerDay.fillna(mt.floor(df['cigsPerDay'].mean()))

df['BPMeds'] = df['BPMeds'].fillna(0)

df['totChol'] = df['totChol'].fillna(234)

df['heartRate'] = df['heartRate'].fillna(70)

df['BMI'] = df['BMI'].fillna(df['BMI'].median())

df['glucose'].value_counts()

df['glucose'] = df['glucose'].fillna(df['glucose'].median())

df.info()

X = df[['male','age','cigsPerDay','totChol','sysBP','glucose']]
Y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test= train_test_split(X, Y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler, Normalizer
norm = Normalizer()
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = linear_model.LogisticRegression()
model.fit(X_train , Y_train)

model.coef_

model.score(X_train , Y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
Y_pred = model.predict(X_test)

confusion_matrix(Y_test,Y_pred)