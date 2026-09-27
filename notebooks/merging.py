import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 7000)

df1 = pd.read_csv('../data/clean_auction1.csv')
df2 = pd.read_csv('../data/clean_auction2.csv')
df3 = pd.read_csv('../data/clean_auction3.csv')
df4 = pd.read_csv('../data/clean_auction4.csv')
df5 = pd.read_csv('../data/clean_auction5.csv')
df6 = pd.read_csv('../data/clean_auction6.csv')

print(f"The shape of the first dataset is {df1.shape}")
print(f"The shape of the second dataset is {df2.shape}")
print(f"The shape of the third dataset is {df3.shape}")
print(f"The shape of the fourth dataset is {df4.shape}")
print(f"The shape of the fifth dataset is {df5.shape}")
print(f"The shape of the sixth dataset is {df6.shape}")

df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

print(f"The shape of the main dataset is {df.shape}")

wdf = df[['subcity', 'district']].drop_duplicates()
wdf.shape

df['district'] = df['district'].ffill()
df['sqm'] = df['sqm'].ffill()

df.to_csv('../data/metadata.csv', index=False)
