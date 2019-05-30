# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:35:21 2019

@author: NIkola
"""


import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

#df = pd.read_csv('../Data/weatherAUS.csv')

df = pd.read_csv('../Data/weatherAUS.csv').sample(frac=1, replace=True, random_state=1)
#df = pd.read_csv('../weatherAUS.csv').sample(n=1000, replace=True, random_state=1)

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

parameters = {
        'n_neighbors' : range(4,5),
        'p' : [2],
        'weights' : ['distance']
        }

knn = GridSearchCV(KNeighborsClassifier(),
                   parameters,
                   cv = 5,
                   scoring = 'f1_macro')

knn.fit(x_train, y_train)

print(knn.cv_results_)
means = knn.cv_results_['mean_test_score']
stds = knn.cv_results_['std_test_score']
params = knn.cv_results_['params']
        
for mean, std, param in zip(means, stds, params):
    print("%0.3f +/- %0.3f za %s" % (mean, 3*std, param))



