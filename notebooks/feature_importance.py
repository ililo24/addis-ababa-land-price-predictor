import joblib
import matplotlib.pyplot as plt
import pandas as pd

pipe = joblib.load('../models/final_model.pkl')

feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()

importances = pipe.named_steps['regressor'].feature_importances_

print(f"Features: {len(feature_names)}, Importances: {len(importances)}")

feat_importances = pd.Series(importances, index=feature_names)
feat_importances.nlargest(10).sort_values().plot(kind='barh', color='skyblue')

plt.title("Top 10 Drivers of Land Price in Addis Ababa")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
