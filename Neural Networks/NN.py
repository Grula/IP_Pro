# -*- coding: utf-8 -*-
"""
Created on Sat May  8 16:24:39 2019

@author: Nikola
"""


import pandas as pd
import numpy as np
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as met
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('../Data/weatherAUS.csv').sample(frac = .2)
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
to_normalize = False
if to_normalize:
    x = pd.DataFrame(prep.MinMaxScaler().fit_transform(df[features]))
else:
    x = df[features]
x.columns = features


y = df['RainTomorrow']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, stratify=y)


"""

NN algorithm

"""
hidden_layer_sizes = []
for i in range(1,2):
    for j in range(6,13):
        hidden_layer_sizes.append((j,)*i)
params = [{'solver': ['sgd','adam'],
           'learning_rate': [ 'adaptive','constant'],
           'activation': [ 'relu', 'logistic'],
           'hidden_layer_sizes': hidden_layer_sizes
           }]


clf = GridSearchCV(MLPClassifier(), params, cv=5)
clf.fit(x_train, y_train)


print("************ [Train] *************")
y_pred =clf.predict(x_train)
print('Precision', met.accuracy_score(y_train, y_pred))
print(met.classification_report(y_train, y_pred))
print()
cnf_matrix = met.confusion_matrix(y_train, y_pred)
print("Confusion Matrix", cnf_matrix, sep="\n")
print("\n")
print("*********************************")

print("************ [Test] *************")
y_pred =clf.predict(x_test)
print('Precision', met.accuracy_score(y_test, y_pred))
print(met.classification_report(y_test, y_pred))
print()
cnf_matrix = met.confusion_matrix(y_test, y_pred)
print("Confusion Matrix", cnf_matrix, sep="\n")
print("\n")
print("*********************************")


print("NN best parameters: \
       \nactivation : %s\
       \nlearning_rate : %s\
       \nsolver : %s" 
      % (clf.best_estimator_.activation, clf.best_estimator_.learning_rate, clf.best_estimator_.solver ))



"""

NN best parameters:        
activation : relu       
learning_rate : constant       
solver : adam

"""











