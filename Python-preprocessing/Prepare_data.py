# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 14:08:12 2019

@author: Nikola
"""


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split


"""
Data specific for dataset
Includes file position
Predictor class, and predictor values (binary)
"""
data_location = '../Data/weatherAUS[Original].csv'
drop_columns = ['Date']
predictor_class = 'RainTomorrow'
pred_true = 'Yes'
pred_false = 'No'
drop_NaN_percent = 0.35



df = pd.read_csv(data_location)
df.drop_duplicates(inplace = True)
df.replace("NA", np.nan, inplace = True)
df.replace("", np.nan, inplace = True)

"""
If data contains more then 35% 
of NaN values it is being droped
"""
data_len = df.shape[0]
for col in df:
    percent = df[col].isna().sum() / data_len
    print("%s & %.2f%s \\\\ \hline" % (col, percent,"\%"))
    if percent > drop_NaN_percent:
        drop_columns.append(col)
df = df.drop(columns = drop_columns)
    


"""
Since our predictor is unbalanced
We are removing excess 'No' from data
first dropping NaN values where predictor is
'No' and then randomly chosing equal amount as 'Yes' 
"""
if False:
    yes = no = 0    
    for row in df[predictor_class]:
        if row == pred_true:
            yes += 1
        else:
            no += 1
    df_yes = df[df[predictor_class] == pred_true].copy(deep = True)
    df_no = df[df[predictor_class] == pred_false].copy(deep = True)
    df_no.dropna(inplace = True)
    df_no = df_no.sample(n = df_yes.shape[0])
    
    
    df = df_yes.append(df_no)
    df = shuffle(df)

df.to_excel(r'../Data/weatherAUS.xlsx')
df.to_csv(r'../Data/weatherAUS.csv')

"""
Creating Train and Test sets 
for modeling algortihms
and saving it in excel format
"""

#features = df.columns[:-1].tolist()
#x = df[features]
#x.columns = features
#y = df[predictor_class]
#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, stratify=y)
#
#x_train.loc[:,predictor_class] = y_train
#x_test.loc[:,predictor_class] = y_test
#
#
#x_train.to_excel(r'../Data/weatherAUS_TrainSET.xlsx')
#x_test.to_excel(r'../Data/weatherAUS_TestSET.xlsx')