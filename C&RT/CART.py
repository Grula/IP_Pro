# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:35:21 2019

@author: NIkola
"""


import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as met
from sklearn import tree

df = pd.read_csv('../weatherAUS.csv')


features = df.columns[1:-1].tolist()

x = df[features]
y = df['RainTomorrow']


# Konverzaciju u Numeric
cities = set(x['Location'])
dct = {}
for a, b in enumerate(cities):
    dct.setdefault(b, list()).append(a)

print(x.head(2))
#for city,val in dct.items():
#    df.loc[city,'Location'] = 0#dct[city][0]

for i in range(len(x)):
    city_name = x.iloc[i,0]
    x.iloc[i,0] = dct[city_name][0]
    if i%100 == 0:
        print("Working on city")

    
#    6,8,9
wind = set(x['WindGustDir'])
dct = {}
for a, b in enumerate(wind):
    dct.setdefault(b, list()).append(a)
for i in range(len(x)):
    wind_1= x.iloc[i,6]
    wind_2= x.iloc[i,8]
    wind_3= x.iloc[i,9]
    if i%100 == 0:
        print("Working on wind")
    x.iloc[i,6] = dct[wind_1][0]
    x.iloc[i,8] = dct[wind_2][0]
    x.iloc[i,9] = dct[wind_3][0]

# 20,21  
rain = set('Yes','No')
dct = {}
for a, b in enumerate(rain):
    dct.setdefault(b, list()).append(a)
    
for i in range(len(x)):
    rain_1= x.iloc[i,20]
    rain_2= x.iloc[i,21]
    x.iloc[i,20] = dct[rain_1][0]
    x.iloc[i,21] = dct[rain_2][0]
    if i%100 == 0:
        print("Working on rain")
    
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)


dt = tree.DecisionTreeClassifier()
dt.fit(x_train, y_train)

#
#f_tree = open('tree.dot', 'w')
#tree.export_graphviz(dt, out_file = f_tree,
#                feature_names = features,
#                class_names = dt.classes_,
#                filled = True,
#                impurity = False)
#f_tree.close
#
print("****** Train *******")
y_pred = dt.predict(x_train)
print("Preciznost", met.accuracy_score(y_train, y_pred)) 
print(pd.DataFrame(met.confusion_matrix(y_train, y_pred),
                   index = dt.classes_,
                   columns = dt.classes_))
print(met.precision_score(y_train, y_pred, average = None))
print(met.classification_report(y_train, y_pred))



print("****** TEST *******")
y_pred = dt.predict(x_test)
print("Preciznost", met.accuracy_score(y_test, y_pred))
print(pd.DataFrame(met.confusion_matrix(y_test, y_pred),
                   index = dt.classes_,
                   columns = dt.classes_))
print(met.classification_report(y_test, y_pred))

