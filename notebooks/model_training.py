import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 4000)

# Prefer the enhanced dataset (it contains the engineered features) when available
base_path = '../data/addis_house_price.csv'
enhanced_path = '../data/addis_house_price_enhanced.csv'
data_path = enhanced_path if os.path.exists(enhanced_path) else base_path
print(f"Loading data from {data_path}")
df = pd.read_csv(data_path)

df = df.dropna(subset=['price_per_sqm', 'lat', 'lon'])
upper_limit = df['price_per_sqm'].quantile(0.99)
df = df[df['price_per_sqm'] <= upper_limit]

df['dist_to_center'] = np.sqrt((df['lat']-9.0)**2 + (df['lon']-38.75)**2)
df['log_sqm'] = np.log1p(df['sqm'])

# Ring features: number of amenities/roads falling in each 1km band
# (difference of the cumulative within-radius counts)
for prefix in ['amenities_within_', 'roads_within_']:
    radii = ['0.5km', '1.0km', '2.0km', '3.0km', '4.0km', '5.0km']
    for r_in, r_out in zip(radii[:-1], radii[1:]):
        col_in, col_out = prefix + r_in, prefix + r_out
        if col_in in df.columns and col_out in df.columns:
            df[f'{prefix}{r_in}_to_{r_out}'] = df[col_out] - df[col_in]

df['subcity'] = df['subcity'].astype('str')
df['district'] = df['district'].astype('str')
df['location_group'] = df['lat'].astype('str') + "_" + df['lon'].astype(str)

if 'neighborhood_cluster' in df.columns:
    df['neighborhood_cluster'] = df['neighborhood_cluster'].astype('str')

df['price_bins'] = pd.qcut(df['price_per_sqm'], q=10, labels=False)

# Grouped 5-fold splits: no location appears in both the train and test side of a fold
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(sgkf.split(df, df['price_bins'], groups=df['location_group']))

print(f"Total rows: {len(df)}")
print(f"Unique locations (groups): {df['location_group'].nunique()}")

drop_cols = ['price_per_sqm', 'price_bins', 'location_group']
X_all = df.drop(columns=drop_cols)
y_all = df['price_per_sqm']
y_all_log = np.log1p(y_all)

# Feature lists (guarded so the script works with both the base and enhanced datasets)
num_cols = ['down_payment_pct', 'sqm', 'log_sqm', 'lon', 'lat', 'dist_to_center',
            'amenities_within_0.5km', 'amenities_within_1.0km', 'amenities_within_2.0km',
            'amenities_within_3.0km', 'amenities_within_4.0km', 'amenities_within_5.0km',
            'roads_within_0.5km', 'roads_within_1.0km', 'roads_within_2.0km',
            'roads_within_3.0km', 'roads_within_4.0km', 'roads_within_5.0km',
            'dist_to_meskel_km', 'dist_to_bole_km', 'amenity_diversity_2km',
            'elevation', 'slope',
            'schools_2km', 'schools_dist_km', 'health_2km', 'health_dist_km',
            'finance_2km', 'finance_dist_km', 'food_2km', 'food_dist_km',
            'markets_2km', 'markets_dist_km',
            'amenities_within_0.5km_to_1.0km', 'amenities_within_1.0km_to_2.0km',
            'amenities_within_2.0km_to_3.0km', 'amenities_within_3.0km_to_4.0km',
            'amenities_within_4.0km_to_5.0km',
            'roads_within_0.5km_to_1.0km', 'roads_within_1.0km_to_2.0km',
            'roads_within_2.0km_to_3.0km', 'roads_within_3.0km_to_4.0km',
            'roads_within_4.0km_to_5.0km']
num_cols = [c for c in num_cols if c in X_all.columns]

cat_cols = ['subcity', 'district']
if 'neighborhood_cluster' in X_all.columns:
    cat_cols.append('neighborhood_cluster')

print(f"Using {len(num_cols)} numeric and {len(cat_cols)} categorical features")

preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), num_cols),
                 ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)]
)

def get_pipeline(model):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

# Stronger hyperparameters than the original run
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_features=0.5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
xgb_model = XGBRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)
lgbm_model = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Stacked ensemble: a Ridge meta-learner learns the optimal blend of the three
# tree models (usually beats a simple average)
stacking_model = StackingRegressor(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('lgbm', lgbm_model)
    ],
    final_estimator=RidgeCV(),
    cv=3
)

# Simple-average ensemble
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
    'Voting Ensemble': em_model,
    'Stacking Ensemble': stacking_model
}

# Honest 5-fold grouped CV for every candidate: out-of-fold predictions over all data
results = {}
best_name, best_r2, best_model = None, -np.inf, None

for name, model in models.items():
    pipe = get_pipeline(model)
    oof_preds_log = cross_val_predict(pipe, X_all, y_all_log, cv=splits)
    oof_preds = np.expm1(oof_preds_log)

    mae = mean_absolute_error(y_all, oof_preds)
    r2 = r2_score(y_all, oof_preds)
    results[name] = (mae, r2)

    print(f"--- {name} ---")
    print(f"CV MAE: {mae:.2f}")
    print(f"CV R2: {r2:.4f}")

    if r2 > best_r2:
        best_name, best_r2, best_model = name, r2, model

print("\n=== 5-fold grouped CV summary ===")
for name, (mae, r2) in results.items():
    print(f"{name}: MAE={mae:.2f}, R2={r2:.4f}")

# Refit the winning model on ALL data so the saved artifact uses every row
print(f"\nRefitting best model ({best_name}) on all data...")
final_pipe = get_pipeline(best_model)
final_pipe.fit(X_all, y_all_log)

# Save the best performing model
joblib.dump(final_pipe, '../models/final_model.pkl')
print(f"\nSaved best model ({best_name}, CV R2={best_r2:.4f}) to '../models/final_model.pkl'")
