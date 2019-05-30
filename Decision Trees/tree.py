# -*- coding: utf-8 -*-
"""
Created on Sat May  7 09:17:22 2019

@author: NIkola
"""


import pandas as pd
import numpy as np
from datetime import datetime
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split
import sklearn.metrics as met
from sklearn import tree


df = pd.read_csv('../Data/weatherAUS.csv')
features = df.columns[1:-1].tolist()

# Pretvaramo datum u dattime
df['Date'] = df['Date'].apply(lambda date : datetime.strptime(date, '%Y-%m-%d').strftime("%m-%d"))

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



# Vrsimo agregaciju nad Godinama
df_aggr = df.groupby(['Date','Location']).max()
df_aggr.reset_index(level=[0,1], inplace = True)


df_aggr = df_aggr.loc[:, df_aggr.columns != 'Date']
df = df.loc[:, df.columns != 'Date']


#Normalizacija
x_1 = pd.DataFrame(prep.MinMaxScaler().fit_transform(df[features]))
x_2 = pd.DataFrame(prep.MinMaxScaler().fit_transform(df_aggr[features]))

#x_1 = df[features]
#x_2 = df_aggr[features]

x_1.columns = features
x_2.columns = features


y_1 = df['RainTomorrow']
y_2 = df_aggr['RainTomorrow']



for x,y in zip((x_1,x_2),(y_1,y_2)):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, stratify=y)
    dt = tree.DecisionTreeClassifier(max_depth = 7)
    dt.fit(x_train, y_train)
    
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

