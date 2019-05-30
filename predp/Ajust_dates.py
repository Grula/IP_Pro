import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('../Data/weatherAUS.csv')

df_dm = df.copy(deep=True) #pd.read_csv('../Data/weatherAUS.csv')

df_dm.replace("", np.nan, inplace = True)

df_dm['Month'] = df_dm['Date'].apply(lambda date : datetime.strptime(date, '%Y-%m-%d').strftime("%m"))
df_dm['Day'] = df['Date'].apply(lambda date : datetime.strptime(date, '%Y-%m-%d').strftime("%d"))
df_dm.drop(columns=['Date'])
df_dm.drop_duplicates(inplace = True)

#df_dm.to_csv(r'weatherAUS_dm.csv')



df_dm.to_excel(r'weatherAUS_dm.xlsx')


#df.to_csv(r'weatherAUS_only_dates.csv')
print("done")