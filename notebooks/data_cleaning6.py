# import the libraries important for data cleaning and preparation
import pandas as pd
import camelot
import warnings

# some tweaks to supress irrelevant warnings and optimize some settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns',  None)

# Convert the pdf into csv
tables = camelot.read_pdf('../data/auction6.pdf', pages='all', flavor='lattice')
combined = pd.concat([table.df for table in tables], ignore_index=True)
combined.to_csv('../data/clean_auction6.csv')

# import the csv file for manipulation and cleaning give the dataset the correct columns name
df = pd.read_csv('../data/clean_auction6.csv')
df.columns = ['one', 'no', 'rank', 'winners', 'price_per_sqm', 'down_payment_pct', 'code', 'district', 'sqm',  'comment']
print(f"The shape of the first dataset is {df.shape}")

# drop irrelevant columns
df.drop(columns=['one', 'no', 'rank', 'winners', 'code', 'comment'], inplace=True)

# drop irrelevant and empty rows, then rearrange the index
df = df.dropna(subset=['price_per_sqm'])
df = df[df['price_per_sqm'].astype(str).str.contains(r'\d', regex=True, na=False)]
df = df.reset_index(drop=True)

# rename the subcity names into the appropriate kinda names
df.loc[0:2, 'subcity'] = 'Kirkos'
df.loc[3:26, 'subcity'] = 'Nefas - Silk Lafto'
df.loc[27:29, 'subcity'] = 'Addis Ketema'
df.loc[30:95, 'subcity'] = 'Kolfe Keranyo'
df.loc[96:209, 'subcity'] = 'Akaki Kality'
df.loc[210:257, 'subcity'] = 'Gulele'
df.loc[258:332, 'subcity'] = 'Lemi Kura'
df.loc[333:, 'subcity'] = 'Yeka'


col_to_move = df.pop(df.columns[4])
df.insert(2, 'subcity', col_to_move)

# inspect the data types and change them to the appropriate ones
df['price_per_sqm'] = df['price_per_sqm'].astype(str).str.replace(',', '').str.strip().astype('float64')
df['down_payment_pct'] = df['down_payment_pct'].astype(str).str.replace('%', '').str.strip().astype('float64')
df['district'] = df['district'].ffill(limit=2).astype('float64')
df['sqm'] = df['sqm'].ffill(limit=2).astype('float64')
print(df.info())

df.to_csv('../data/clean_auction6.csv', index=False)
