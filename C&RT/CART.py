# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:35:21 2019

@author: NIkola
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as met
from sklearn import tree

df = pd.read_csv('../weatherAUS.csv')
#df = pd.read_csv('../weatherAUS.csv').sample(frac=0.5, replace=True, random_state=1)
#df = pd.read_csv('../weatherAUS.csv').sample(n=1000, replace=True, random_state=1)

features = df.columns[1:-1].tolist()


df.drop_duplicates(inplace = True)
df.replace("", np.nan, inplace = True)
#df.dropna(inplace = True)

print(features)
i = 0
median = df.loc[:,'Rainfall'].median()
for elem in df['RainToday']:
    if elem not in("Yes","No"):
        rainfall = df.iloc[i,4]
#        print("This is element %s" % elem)
#        print("This is rainfall %s" % rainfall)
        if rainfall != np.nan:
            continue
            print(df.iloc[21 , i])
            df.iloc[i , 21] = median if rainfall >= 1  else 0
    i += 1

# Konverzija podataka u numericke
cities = list(set(df['Location']))
df.replace(cities, list(range(0,len(cities))), inplace = True)
    
wind = list(set(df['WindGustDir']))
df.replace(wind, list(range(0,len(wind))), inplace = True)

rain = ['Yes','No']
df.replace(rain,[0,1], inplace = True)
    

x = df[features]
y = df['RainTomorrow']




x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)
dt = tree.DecisionTreeClassifier()
dt.fit(x_train, y_train)


print("****** Train *******")
y_pred = dt.predict(x_train)
print("Precision", met.accuracy_score(y_train, y_pred)) 
print(pd.DataFrame(met.confusion_matrix(y_train, y_pred),
                   index = dt.classes_,
                   columns = dt.classes_))
print(met.precision_score(y_train, y_pred))
print(met.classification_report(y_train, y_pred))



print("****** TEST *******")
y_pred = dt.predict(x_test)
print("Precision", met.accuracy_score(y_test, y_pred))
print(pd.DataFrame(met.confusion_matrix(y_test, y_pred),
                   index = dt.classes_,
                   columns = dt.classes_))
print(met.classification_report(y_test, y_pred))

