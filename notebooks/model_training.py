import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from category_encoders import TargetEncoder
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 4000)
df = pd.read_csv('../data/addis_house_price.csv')

df = df.dropna(subset=['price_per_sqm', 'lat', 'lon'])
upper_limit = df['price_per_sqm'].quantile(0.99)
df = df[df['price_per_sqm'] <= upper_limit]

df['dist_to_center'] = np.sqrt((df['lat']-9.0)**2 + (df['lon']-38.75)**2)

df['subcity'] = df['subcity'].astype('str')
df['district'] = df['district'].astype('str')
df['location_group'] = df['lat'].astype('str') + "_" + df['lon'].astype(str)

df['price_bins'] = pd.qcut(df['price_per_sqm'], q=10, labels=False)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

train_idx, test_idx = next(sgkf.split(df, df['price_bins'], groups = df['location_group']))

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

print(f"Total rows: {len(df)}")
print(f"Train set: {len(train_df)} rows")
print(f"Test set: {len(test_df)} rows")

overlap = set(train_df['location_group']).intersection(set(test_df['location_group']))
print(f"Number of overlapping locations: {len(overlap)}")

X = train_df.drop(columns=['price_per_sqm', 'price_bins', 'location_group'])
y = train_df['price_per_sqm']

X_test = test_df.drop(columns=['price_per_sqm', 'price_bins', 'location_group'])
y_test = test_df['price_per_sqm']

rf_model = RandomForestRegressor(random_state=42)
xgb_model = XGBRegressor(random_state=42)
lgbm_model = LGBMRegressor(random_state=42, verbose=-1)

# The Ensemble
em_model = VotingRegressor(
    estimators=[
        ('Random Forest', rf_model),
        ('XGBoost', xgb_model),
        ('LightGBM', lgbm_model)
    ]
)

models = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'LightGBM': lgbm_model,
    'Ensemble Model': em_model
}

num_cols = ['down_payment_pct', 'sqm', 'lon', 'lat', 'amenities_within_0.5km', 'amenities_within_1.0km','amenities_within_2.0km', 'amenities_within_3.0km', 'amenities_within_4.0km', 'amenities_within_5.0km', 'roads_within_0.5km', 'roads_within_1.0km', 'roads_within_2.0km', 'roads_within_3.0km', 'roads_within_4.0km', 'roads_within_5.0km']
cat_cols = ['subcity', 'district']

preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), num_cols),
                 ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)]
)

def get_pipeline(model):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

results = {}
models = {
    'RF': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGB': XGBRegressor(n_estimators=100, learning_rate=0.05),
    'LGBM': LGBMRegressor(verbose=-1)
}

y_train_log = np.log1p(y)
y_test_log = np.log1p(y_test)

for name, model in models.items():
    pipe = get_pipeline(model)
    pipe.fit(X, y_train_log) 
    
    preds_log = pipe.predict(X_test)
    preds = np.expm1(preds_log) 
    
    print(f"--- {name} ---")
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    print(f"R2: {r2_score(y_test, preds):.4f}")

import joblib
joblib.dump(pipe, '../models/final_model.pkl')
