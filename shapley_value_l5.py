import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

# 1) Ensure output directories exist
os.makedirs('level_5_visualizations', exist_ok=True)
os.makedirs('primary_models', exist_ok=True)

# 2) Load and prepare data
df = pd.read_csv('data/data_level_5.csv')
df['log_price']             = np.log(df['price_per_sqm'])
df['log_area']              = np.log(df['area'])
df['log_distance_capital']  = np.log(df['distance_capital'])

predictors = [
    'log_area', 'log_distance_capital'
]
X = df[predictors]
y = df['log_price']

# 3) Standardize features for comparability
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=predictors)

# 4) Fit equivalent sklearn linear model
lr = LinearRegression()
lr.fit(X_scaled, y)
R2 = lr.score(X_scaled, y)

# 5) Compute SHAP values - using the recommended masker approach
masker = shap.maskers.Independent(X_scaled)
explainer = shap.LinearExplainer(lr, masker)
shap_values = explainer.shap_values(X_scaled)

# 6) Compute metrics
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': predictors,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)
importance_df['pct_importance'] = importance_df['mean_abs_shap'] / importance_df['mean_abs_shap'].sum()
importance_df['explained_variance_share'] = importance_df['pct_importance'] * R2

# 7) Save metrics to CSV
importance_df.to_csv('primary_models/shap_importance_l5.csv', index=False)

# 8) Plot and save bar chart
plt.figure(figsize=(8,6))
plt.barh(importance_df['feature'], importance_df['mean_abs_shap'])
plt.xlabel('Mean |Shapley value|')
plt.title('Global Feature Importance (SHAP)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('level_5_visualizations/shap_importance.png')
plt.close()

# 9) Create and save summary beeswarm plot
shap.summary_plot(shap_values, X_scaled_df, plot_type="dot", show=False)
plt.title('SHAP Summary Plot')
plt.savefig('level_5_visualizations/shap_summary.png', bbox_inches='tight')
plt.close()
