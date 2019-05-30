# -*- coding: utf-8 -*-
"""
Created on Sat May  8 16:24:39 2019

@author: NIkola
"""


import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as met
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('../Data/weatherAUS.csv')

features = df.columns[1:-1].tolist()

# Pretvaramo datum u dattime
df['Date'] = df['Date'].apply(lambda date : datetime.strptime(date, '%Y-%m-%d').strftime("%m-%d"))

#df.drop_duplicates(inplace = True)
df.replace("", np.nan, inplace = True)

#Skidanje elemenata van granica
for col in features:
    val = df[col].head(1).values[0]     
    if isinstance(val, (np.float64)):
        q1 = df[col].quantile(0.25)
        q2 = df[col].quantile(0.5)
        q3 = df[col].quantile(0.75)
        ext = [q1-3*(q3-q1),q3+3*(q3-q1)]
        df.drop( df[(df[col]<ext[0]) | (df[col]>ext[1])].index, inplace = True)
    else:
        continue

# Konverzija podataka u numericke
cities = list(set(df['Location']))
df.replace(cities, list(range(0,len(cities))), inplace = True)
    
wind = list(set(df['WindGustDir']))
df.replace(wind, list(range(0,len(wind))), inplace = True)

rain = ['Yes','No']
df.replace(rain, [0,1], inplace = True)



#Korelacija
df_corel = df.copy(deep = True)
df_corel.dropna(inplace = True)
corel = df_corel.corr(method='pearson')
corel_features = corel.columns[:].tolist()
elements_in_corelation = {}
for col1 in corel_features:    
    for col2 in corel_features:
       if col1 != col2 and abs(corel.loc[col1,col2]) >= 0.8 and not elements_in_corelation.get(col2)  :
           elements_in_corelation[col1] = col2
        
df = df.drop(columns = list(elements_in_corelation.keys()))
features = df.columns[1:-1].tolist()

x = df[features]
x.columns = features
y = df['RainTomorrow']


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, stratify=y)



hidden_layer_sizes = []
for i in range(1,2):
    for j in range(9,12):
        hidden_layer_sizes.append((j,)*i)


params = [{'solver': ['sgd'],
           'learning_rate': [ 'adaptive'],
           'activation': [ 'relu'],
           'hidden_layer_sizes': hidden_layer_sizes
           #'max_iter': [600]

           }]

clf = GridSearchCV(MLPClassifier(), params, cv=5)

clf.fit(x_train, y_train)

#print("Najbolji parametri:")
#print(clf.best_params_)
#print()

print("Izvestaj za trening skup:")
y_pred =clf.predict(x_train)

print('Preciznost', met.accuracy_score(y_train, y_pred))
print(met.classification_report(y_train, y_pred))
print()

cnf_matrix = met.confusion_matrix(y_train, y_pred)
print("Matrica konfuzije", cnf_matrix, sep="\n")
print("\n")

print("Izvestaj za test skup:")
y_pred =clf.predict(x_test)
print('Preciznost', met.accuracy_score(y_test, y_pred))
print(met.classification_report(y_test, y_pred))
print()

cnf_matrix = met.confusion_matrix(y_test, y_pred)
print("Matrica konfuzije", cnf_matrix, sep="\n")
print("\n")















