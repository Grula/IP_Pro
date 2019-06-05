# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:35:21 2019

@author: Nikola
"""


import pandas as pd
import numpy as np
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as met


df = pd.read_csv('../Data/weatherAUS.csv').sample(frac = .5)
features = df.columns[1:-1].tolist()

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
        q2 = df[col].quantile(0.5)
        q3 = df[col].quantile(0.75)
        ext = [q1-1.5*(q3-q1),q3+1.5*(q3-q1)]
        df.drop( df[(df[col]<ext[0]) | (df[col]>ext[1])].index, inplace = True)

"""
Conversion of String elements into numeric
"""
string_elemets = ['Location','WindGustDir','RainTomorrow']
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
print(elements_in_corelation)
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
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, stratify=y)

"""

KNN algorithm

"""
parameters = {
        'n_neighbors' : range(3,6),
        'p' : [1,2],
        'weights' : ['distance']
        }

knn = GridSearchCV(KNeighborsClassifier(), parameters, cv = 5, scoring = 'f1_macro')
knn.fit(x_train, y_train)


means = knn.cv_results_['mean_test_score']
stds = knn.cv_results_['std_test_score']
params = knn.cv_results_['params']        
for mean, std, param in zip(means, stds, params):
    print("%0.3f +/- %0.3f for %s" % (mean, 3*std, param))

print("************ [Train] *************")
y_pred =knn.predict(x_train)
print('Precision', met.accuracy_score(y_train, y_pred))
print(met.classification_report(y_train, y_pred))
print()
cnf_matrix = met.confusion_matrix(y_train, y_pred)
print("Confusion Matrix", cnf_matrix, sep="\n")
print("\n")
print("*********************************")

print("************ [Test] *************")
y_pred =knn.predict(x_test)
print('Precision', met.accuracy_score(y_test, y_pred))
print(met.classification_report(y_test, y_pred))
print()
cnf_matrix = met.confusion_matrix(y_test, y_pred)
print("Confusion Matrix", cnf_matrix, sep="\n")
print("\n")
print("*********************************")

print("KNN with best score : \n%s" % (knn.best_estimator_))

"""
KNN with best score : 
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=5, p=1,
           weights='distance')
"""
