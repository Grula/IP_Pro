# -*- coding: utf-8 -*-
"""
Created on Sat May 5 14:15:41 2019

@author: Nikola
"""


import pandas as pd
import numpy as np
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as met
from sklearn import svm

df = pd.read_csv('../Data/weatherAUS.csv').sample(frac = .2)
features = df.columns[1:-1].tolist()
print(features)

"""
Replacing NaN values
integer type - change to column mean
string type - change to most occuring string
"""
for col in features:
    if df[col].isna().sum() != 0:
        if isinstance(df[col].mode()[0], (np.float64)):
            df[col].replace(np.nan, df[col].mean(), inplace = True)
        else:
            df[col].replace(np.nan, df[col].mode()[0], inplace = True)

"""
Removing elements outside the boundaries
"""
for col in features:
    val = df[col].head(1).values[0]     
    if isinstance(val, (np.float64)):
        q1 = df[col].quantile(0.25)
#        q2 = df[col].quantile(0.5)
        q3 = df[col].quantile(0.75)
        ext = [q1-1.5*(q3-q1),q3+1.5*(q3-q1)]
        df.drop( df[(df[col]<ext[0]) | (df[col]>ext[1])].index, inplace = True)

"""
Conversion of String elements into numeric
"""
string_elemets = ['Location','WindGustDir','RainToday']
for element in string_elemets:
    list_of_elements = list(set(df[element]))
    df.replace(list_of_elements, list(range(0,len(list_of_elements))), inplace = True)

"""
Finding corelation between elemts
to reduce data size
"""
df_corel = df.copy(deep = True)
corel = df_corel.corr( method='pearson' )
corel_features = corel.columns[:-1].tolist()
elements_in_corelation = {}
for col1 in corel_features:    
    for col2 in corel_features:
       if col1 != col2 and abs(corel.loc[col1,col2]) >= 0.8 and not elements_in_corelation.get(col2)  :
           elements_in_corelation[col1] = col2
df = df.drop(columns = list(elements_in_corelation.keys()))
features = df.columns[:-1].tolist()

"""
Nomrlazing data
"""
to_normalize = True
if to_normalize:
    x = pd.DataFrame(prep.MinMaxScaler().fit_transform(df[features]))
else:
    x = df[features]
x.columns = features



y = df['RainTomorrow']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, stratify = y)

"""

SVM algorithm

"""

parameters = [
            {
                'C' : [1,5],
                'kernel' : ['poly'],
                'degree' : [1,3],
                'gamma' : [0.5,1],
                'coef0' : [.5]
            },
            {
                'C' : [10],
                'kernel' : ['rbf'],
                'coef0' : [.5,1]
            }
]
clv = GridSearchCV(svm.SVC(), parameters, cv=3, scoring='f1_macro')
clv.fit(x_train, y_train)

for mean, param in zip(clv.cv_results_['mean_test_score'], clv.cv_results_['params']):
    print("%0.3f za %s" % (mean, param))
    
print("************ [Train] *************")
y_pred =clv.predict(x_train)
print('Precision', met.accuracy_score(y_train, y_pred))
print(met.classification_report(y_train, y_pred))
print()
cnf_matrix = met.confusion_matrix(y_train, y_pred)
print("Confusion Matrix", cnf_matrix, sep="\n")
print("*********************************")

print("************ [Test] *************")
y_pred =clv.predict(x_test)
print('Precision', met.accuracy_score(y_test, y_pred))
print(met.classification_report(y_test, y_pred))
print()
cnf_matrix = met.confusion_matrix(y_test, y_pred)
print("Confusion Matrix", cnf_matrix, sep="\n")
print("*********************************")

print("SVM best parameters: \
       \nactivation : %s\
       \nlearning_rate : %s\
       \nsolver : %s" 
      % (clv.best_estimator_.C, clv.best_estimator_.kernel, clv.best_estimator_.coef0 ))
print(clv.best_estimator_.n_support_)

