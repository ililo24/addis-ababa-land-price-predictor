
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    
    df['subcity'] = df['subcity'].astype('str')
    df['district'] = df['district'].astype('str')
    df['neighborhood_cluster'] = df['neighborhood_cluster'].astype('str')
    df['location_group'] = df['lat'].astype('str') + "_" + df['lon'].astype(str)
    
    df['price_bins'] = pd.qcut(df['price_per_sqm'], q=10, labels=False)
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(df, df['price_bins'], groups = df['location_group']))
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    X = train_df.drop(columns=['price_per_sqm', 'price_bins', 'location_group'])
    y = train_df['price_per_sqm']
    X_test = test_df.drop(columns=['price_per_sqm', 'price_bins', 'location_group'])
    y_test = test_df['price_per_sqm']
    
    # Define feature groups
    old_num_cols = ['down_payment_pct', 'sqm', 'lon', 'lat', 
                    'amenities_within_0.5km', 'amenities_within_1.0km', 'amenities_within_2.0km', 
                    'amenities_within_3.0km', 'amenities_within_4.0km', 'amenities_within_5.0km', 
                    'roads_within_0.5km', 'roads_within_1.0km', 'roads_within_2.0km', 
                    'roads_within_3.0km', 'roads_within_4.0km', 'roads_within_5.0km']
    
    new_num_cols = ['dist_to_meskel_km', 'dist_to_bole_km', 'amenity_diversity_2km', 'dist_to_center']
    
    num_cols = old_num_cols + new_num_cols
    cat_cols = ['subcity', 'district', 'neighborhood_cluster']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    print("Training model...")
    y_train_log = np.log1p(y)
    pipe.fit(X, y_train_log)
    
    print("Evaluating model...")
    preds_log = pipe.predict(X_test)
    preds = np.expm1(preds_log)
    
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Enhanced Model - MAE: {mae:.2f}, R2: {r2:.4f}")
    
    # Feature Importance
    print("\nTop 15 Most Important Features:")
    feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()
    importances = pipe.named_steps['regressor'].feature_importances_
    feat_importances = pd.Series(importances, index=feature_names)
    print(feat_importances.nlargest(15))
    
    # Save model
    model_path = 'models/final_model_enhanced.pkl'
    joblib.dump(pipe, model_path)
    print(f"Model saved to {model_path}")
    
    return mae, r2

if __name__ == "__main__":
    train_enhanced_model('data/addis_house_price_enhanced.csv')
