import pandas as pd



df = pd.read_csv(r'weatherAUS.csv')

for date in df['Date']:
    df[date] = date.split('-')[1]

#df.to_csv()
print("done")