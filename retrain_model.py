
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

def train_enhanced_model(file_path):
    print(f"Loading enhanced data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Preprocessing identical to notebook
    df = df.dropna(subset=['price_per_sqm', 'lat', 'lon'])
    upper_limit = df['price_per_sqm'].quantile(0.99)
    df = df[df['price_per_sqm'] <= upper_limit]
    
    # Add dist_to_center if not present
    if 'dist_to_center' not in df.columns:
        df['dist_to_center'] = np.sqrt((df['lat']-9.0)**2 + (df['lon']-38.75)**2)
    
    # log_sqm captures the nonlinear small-plot premium better than raw sqm
    if 'sqm' in df.columns:
        df['log_sqm'] = np.log1p(df['sqm'])
    
    # Ring features: number of amenities/roads falling in each 1km band
    for prefix in ['amenities_within_', 'roads_within_']:
        radii = ['0.5km', '1.0km', '2.0km', '3.0km', '4.0km', '5.0km']
        for r_in, r_out in zip(radii[:-1], radii[1:]):
            col_in, col_out = prefix + r_in, prefix + r_out
            if col_in in df.columns and col_out in df.columns:
                df[f'{prefix}{r_in}_to_{r_out}'] = df[col_out] - df[col_in]
    
    df['subcity'] = df['subcity'].astype('str')
    df['district'] = df['district'].astype('str')
    if 'neighborhood_cluster' in df.columns:
        df['neighborhood_cluster'] = df['neighborhood_cluster'].astype('str')
    df['location_group'] = df['lat'].astype('str') + "_" + df['lon'].astype(str)
    
    df['price_bins'] = pd.qcut(df['price_per_sqm'], q=10, labels=False)
    
    # Grouped 5-fold splits: no location appears in both the train and test side of a fold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(df, df['price_bins'], groups = df['location_group']))
    
    X_all = df.drop(columns=['price_per_sqm', 'price_bins', 'location_group'])
    y_all = df['price_per_sqm']
    y_all_log = np.log1p(y_all)
    
    # Define feature groups (guarded so missing engineered columns don't break the run)
    old_num_cols = ['down_payment_pct', 'sqm', 'log_sqm', 'lon', 'lat', 
                    'amenities_within_0.5km', 'amenities_within_1.0km', 'amenities_within_2.0km', 
                    'amenities_within_3.0km', 'amenities_within_4.0km', 'amenities_within_5.0km', 
                    'roads_within_0.5km', 'roads_within_1.0km', 'roads_within_2.0km', 
                    'roads_within_3.0km', 'roads_within_4.0km', 'roads_within_5.0km']
    
    new_num_cols = ['dist_to_meskel_km', 'dist_to_bole_km', 'amenity_diversity_2km', 'dist_to_center', 'elevation', 'slope',
                    'schools_2km', 'schools_dist_km', 'health_2km', 'health_dist_km',
                    'finance_2km', 'finance_dist_km', 'food_2km', 'food_dist_km',
                    'markets_2km', 'markets_dist_km',
                    'amenities_within_0.5km_to_1.0km', 'amenities_within_1.0km_to_2.0km',
                    'amenities_within_2.0km_to_3.0km', 'amenities_within_3.0km_to_4.0km',
                    'amenities_within_4.0km_to_5.0km',
                    'roads_within_0.5km_to_1.0km', 'roads_within_1.0km_to_2.0km',
                    'roads_within_2.0km_to_3.0km', 'roads_within_3.0km_to_4.0km',
                    'roads_within_4.0km_to_5.0km']
    
    num_cols = [c for c in old_num_cols + new_num_cols if c in df.columns]
    cat_cols = ['subcity', 'district']
    if 'neighborhood_cluster' in df.columns:
        cat_cols.append('neighborhood_cluster')
    
    print(f"Using {len(num_cols)} numeric and {len(cat_cols)} categorical features")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    
    def get_pipeline(model):
        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
    
    # Tuned tree models
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
    
    # Stacked ensemble (Ridge meta-learner) and simple-average ensemble
    stacking_model = StackingRegressor(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lgbm', lgbm_model)
        ],
        final_estimator=RidgeCV(),
        cv=3
    )
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
    print("Running 5-fold grouped CV for all candidate models...")
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
    
    # Feature Importance (averaged over the tree models in the ensemble)
    print("\nTop 15 Most Important Features:")
    feature_names = final_pipe.named_steps['preprocessor'].get_feature_names_out()
    regressor = final_pipe.named_steps['regressor']
    if hasattr(regressor, 'named_estimators_'):
        importances = np.zeros(len(feature_names))
        n_tree_models = 0
        for est_name, est in regressor.named_estimators_.items():
            if hasattr(est, 'feature_importances_'):
                importances += est.feature_importances_
                n_tree_models += 1
        if n_tree_models > 0:
            feat_importances = pd.Series(importances / n_tree_models, index=feature_names)
            print(feat_importances.nlargest(15))
        else:
            print("Tree-based feature importances not available for this model.")
    elif hasattr(regressor, 'feature_importances_'):
        feat_importances = pd.Series(regressor.feature_importances_, index=feature_names)
        print(feat_importances.nlargest(15))
    else:
        print("Tree-based feature importances not available for this model.")
    
    # Save model
    model_path = 'models/final_model_enhanced.pkl'
    joblib.dump(final_pipe, model_path)
    print(f"Model saved to {model_path}")
    
    return best_mae, best_r2

if __name__ == "__main__":
    train_enhanced_model('data/addis_house_price_enhanced.csv')
