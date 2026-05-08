#Addis Ababa Land Price Intelligence

##Executive Summary

This project provides a data-driven approach to estimating land values in Addis Ababa. By leveraging historical auction data and machine learning, this tool helps investors, developers, and researchers navigate the complex real estate market in Ethiopia's capital.

The primary goal was to move beyond simple property listings and build a robust model capable of handling the high-cardinality and spatial nuances of the Addis Ababa land market.

##Key Insights
* **Location Location Location:** My model identifies dist_to_center, amenity and road counts within a district/woreda (calculated via Haversine geometry) as the most critical determinant of land value.
* **Market Skew:** I discovered that the Addis land market follows a "long-tail" distribution. Standard linear models failed to capture this, so I implemented Target Encoding and Log-Transformation to stabilize the variance, achieving an r2_score of ~0.34.
* **Transparency:** By using feature importance analysis, I demystified the "black box" of real estate pricing, providing clear insights into which factors (subcity, square meters, amenities) drive premiums.

##Tech Stacks
* **Language:** Python 3.12
* **Core Libraries:** Pandas, NumPy, Scikit-learn
* **Modeling:** XGBoost, LightGBM, RandomForestRegressor
* **Preprocessing:** TargetEncoder (to handle high-cardinality categorical variables)
* **Environment:** Jupyter Notebooks & Scikit-learn Pipelines

##Project Structure
```text

auction/
├── data/               # Raw auction datasets
├── models/             # Serialized best-performing models (.pkl)
├── notebooks/          # Step-by-step analysis
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_interpretability.ipynb
├── RAG/                # (Coming Soon) Legal code knowledge base
└── README.md           # Project documentation
```
##Methodology

1. Data Cleaning: Handled outliers and missing values to ensure model stability.
2. Feature Engineering: Engineered spatial features, such as distance-to-center and proximity to key infrastructure, to capture latent location value.
3. Modeling Strategy: Implemented a pipeline-based approach to ensure no data leakage occurred during training, using StratifiedGroupKFold to group by geographic location.
4. Explainability: Transitioned from black-box predictions to transparent decision-making by analyzing feature importance and residual patterns.

##Future Roadmap

* **RAG Implementation:** Integrating Addis Ababa’s land lease regulations into a Chatbot using LangChain to allow users to query legal and zoning requirements.
* **Spatial Heatmaps:** Adding interactive Folium maps to visualize price clusters across the city.
* **Enhanced Features:** Incorporating property age, soil type and zoning classification to increase model accuracy (r2_score > 0.6)

Created by Ililo Altaye – Aspiring Data Scientist.
