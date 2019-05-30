# -*- coding: utf-8 -*-
"""
Created on Sat May 5 14:15:41 2019

@author: NIkola
"""


import pandas as pd
import numpy as np
from datetime import datetime
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm, metrics


df = pd.read_csv('../Data/weatherAUS.csv').sample(frac=.3, replace=True, random_state=1)
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


df = df.loc[:, df.columns != 'Date']


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
        
print(list(elements_in_corelation.keys()))
df = df.drop(columns = list(elements_in_corelation.keys()))
features = df.columns[1:-1].tolist()
print(features)

#Normalizacija
x = pd.DataFrame(prep.MinMaxScaler().fit_transform(df[features]))
x.columns = features


y = df['RainTomorrow']


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, stratify=y)
#parameters = [{
#                'C' : [0.1, 0.5, 1, 10, 20],
#                'kernel' : ['linear'],
#              },
#              {
#                'C' : [0.1, 0.5, 1, 10, 20],
#                'kernel' : ['poly'],
#                'degree' : [1,2,3,4,5],
#                'gamma' : [0.1, 0.2, 0.5, 1],
#                'coef0' : [0, .2, .3, .4, .5, 1]
#              },
#              {
#                'C' : [0.1, 0.5, 1, 10, 20],
#                'kernel' : ['rbf'],
#                'coef0' : [0, .2, .5, 1]
#             }]

parameters = [
            {
                'C' : [ 0.5],
                'kernel' : ['poly'],
                'degree' : [5],
                'gamma' : [0.5, 1],
                'coef0' : [0]
            },
              {
                'C' : [20, 30, 40],
                'kernel' : ['rbf'],
                'coef0' : [.5, 1]
             }]
clv = GridSearchCV(svm.SVC(), parameters, cv=5, scoring='f1_macro')
clv.fit(x_train, y_train)

for mean, param in zip(clv.cv_results_['mean_test_score'], clv.cv_results_['params']):
    print("%0.3f za %s" % (mean, param))
    
print("Trening skup")
y_pred = clv.predict(x_train)
print(metrics.accuracy_score(y_train, y_pred))
print(metrics.classification_report(y_train, y_pred))
    
print("Test skup")
y_pred = clv.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))


print(clv.best_estimator_.n_support_)

