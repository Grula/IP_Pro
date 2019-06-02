# -*- coding: utf-8 -*-
"""
Created on Sat May  7 09:17:22 2019

@author: Nikola
"""


import pandas as pd
import numpy as np
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as met
from sklearn import tree

df = pd.read_csv('../Data/weatherAUS.csv')
features = df.columns[2:-1].tolist()

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
string_elemets = ['Location','WindGustDir','RainToday']
for element in string_elemets:
    list_of_elements = list(set(df[element]))
    df.replace(list_of_elements, list(range(0,len(list_of_elements))), inplace = True)
    
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


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)



"""

Decision Tree

"""
parameters = {
        'max_depth' : range(4,12)
        }

tree = GridSearchCV(tree.DecisionTreeClassifier(),
                   parameters,
                   cv = 10,
                   scoring = 'f1_macro')
tree.fit(x_train, y_train)

print("************ [Train] *************")
y_pred = tree.predict(x_train)
print('Precision', met.accuracy_score(y_train, y_pred))
print(met.classification_report(y_train, y_pred))
print()
cnf_matrix = met.confusion_matrix(y_train, y_pred)
print("Confusion Matrix", cnf_matrix, sep="\n")
print("\n")
print("*********************************")

print("************ [Test] *************")
y_pred = tree.predict(x_test)
print('Precision', met.accuracy_score(y_test, y_pred))
print(met.classification_report(y_test, y_pred))
print()
cnf_matrix = met.confusion_matrix(y_test, y_pred)
print("Confusion Matrix", cnf_matrix, sep="\n")
print("\n")
print("*********************************")

print("Tree depth with best score : %s" % (tree.best_estimator_.max_depth))

"""

"""
