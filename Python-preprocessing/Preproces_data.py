# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 14:08:12 2019

@author: Nikola
"""

import pandas as pd
import numpy as np

def data_preprocessing(df , features, string_elemets, corelation = False):
    """
    Removing elements outside the boundaries
    """
    for col in features:
        val = df[col].head(1).values[0]     
        if isinstance(val, (np.float64)):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            ext = [q1-1.5*(q3-q1),q3+1.5*(q3-q1)]
            df.drop( df[(df[col]<ext[0]) | (df[col]>ext[1])].index, inplace = True)
    
    """
    Conversion of String elements into numeric
    """
    for element in string_elemets:
        list_of_elements = list(set(df[element]))
        df.replace(list_of_elements, list(range(0,len(list_of_elements))), inplace = True)
        
    """
    Corelation between atributes.
    Atributes with corelation greather than 0.8
    are removed from dataset
    """
    if corelation:
        df_corel = df.copy(deep = True)
        corel = df_corel.corr( method='pearson' )
        corel_features = corel.columns[:-1].tolist()
        elements_in_corelation = {}
        for col1 in corel_features:    
            for col2 in corel_features:
               if col1 != col2 and abs(corel.loc[col1,col2]) >= 0.8 and not elements_in_corelation.get(col2)  :
                   elements_in_corelation[col1] = col2
        df = df.drop(columns = list(elements_in_corelation.keys()))
        
    return df
            