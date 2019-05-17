# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:35:21 2019

@author: NIkola
"""


import pandas as pd
import numpy as np
from datetime import datetime
import copy
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split
import sklearn.metrics as met
from sklearn import tree



def prune(tree):
    tree = copy.deepcopy(tree)
    dat = tree.tree_
    nodes = range(0, dat.node_count)
    ls = dat.children_left
    rs = dat.children_right
    classes = [[list(e).index(max(e)) for e in v] for v in dat.value]

    leaves = [(ls[i] == rs[i]) for i in nodes]

    LEAF = -1
    for i in reversed(nodes):
        if leaves[i]:
            continue
        if leaves[ls[i]] and leaves[rs[i]] and classes[ls[i]] == classes[rs[i]]:
            ls[i] = rs[i] = LEAF
            leaves[i] = True
    return tree



df = pd.read_csv('../weatherAUS.csv')
#df = pd.read_csv('../weatherAUS.csv').sample(frac=0.5, replace=True, random_state=1)
#df = pd.read_csv('../weatherAUS.csv').sample(n=1000, replace=True, random_state=1)

features = df.columns[1:-1].tolist()

# Pretvaramo datum u dattime
df['Date'] = df['Date'].apply(lambda date : datetime.strptime(date, '%Y-%m-%d').strftime("%m-%d"))

df.drop_duplicates(inplace = True)
df.replace("", np.nan, inplace = True)

df.dropna(inplace = True)
#df = df.fillna(df.mode().iloc[0])


#Skidanje Anomalija
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

# Vrsimo agregaciju nad Godinama
df = df.groupby(['Date','Location']).max()
df.reset_index(level=[0,1], inplace = True)


df = df.loc[:, df.columns != 'Date']
#Normalizacija
x = pd.DataFrame(prep.MinMaxScaler().fit_transform(df[features]))
x.columns = features


# Trazimo korelaciju izmedju elemenata
# Kod drveta odlucivanja ne moramo to da radimo
# Podaci nemaju previsok broj atributa
#corel = df.corr(method='pearson')
#corel_features = corel.columns[:].tolist()
#elements_in_corelation = {}
#for col1 in corel_features:    
#    for col2 in corel_features:
#       if col1 != col2 and abs(corel.loc[col1,col2]) >= 0.8 and not elements_in_corelation.get(col2)  :
#           elements_in_corelation[col1] = col2


y = df['RainTomorrow']


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, stratify=y)
dt = tree.DecisionTreeClassifier(max_depth = 7)
dt.fit(x_train, y_train)

#dt = prune(dt)

print("****** Train *******")
y_pred = dt.predict(x_train)
print("Precision", met.accuracy_score(y_train, y_pred)) 
print(pd.DataFrame(met.confusion_matrix(y_train, y_pred),
                   index = dt.classes_,
                   columns = dt.classes_))
print(met.precision_score(y_train, y_pred))
print(met.classification_report(y_train, y_pred))



print("****** Test *******")
y_pred = dt.predict(x_test)
print("Precision", met.accuracy_score(y_test, y_pred))
print(pd.DataFrame(met.confusion_matrix(y_test, y_pred),
                   index = dt.classes_,
                   columns = dt.classes_))
print(met.classification_report(y_test, y_pred))

