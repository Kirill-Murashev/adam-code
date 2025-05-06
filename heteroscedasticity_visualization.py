import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import os

# Ensure output directory exists
os.makedirs('level_1_visualizations', exist_ok=True)

# 1) Load data and fit model
df = pd.read_csv('data/data_level_1.csv')
df['log_price']             = np.log(df['price_per_sqm'])
df['log_area']              = np.log(df['area'])
df['log_distance_capital']  = np.log(df['distance_capital'])
df['log_distance_elevator'] = np.log(df['distance_elevator'])
df['log_crop_yield']        = np.log(df['crop_yield'])

formula = (
    "log_price ~ "
    "log_area + log_distance_capital + log_distance_elevator + log_crop_yield + "
    "access + coast_line + ownership + simple_shape + is_marked + "
    "is_ab + is_cd + is_north_forest_steppe + is_south_forest_steppe + is_steppe"
)
model = smf.ols(formula, data=df).fit()

# 2) Calculate fitted values and residuals
fitted = model.fittedvalues
resid = model.resid
influence = model.get_influence()
std_resid = influence.resid_studentized_internal

# 3) Spread-Level plot (Residuals vs. |Fitted|)
plt.figure(figsize=(8,5))
plt.scatter(np.abs(fitted), resid, alpha=0.6, edgecolor='k')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('|Fitted values|')
plt.ylabel('Residuals')
plt.title('Spread-Level Plot: Residuals vs. |Fitted|')
plt.tight_layout()
plt.savefig('level_1_visualizations/spread_level_plot.png')
plt.close()

# 4) Scale-Location plot (sqrt(|Std Residuals|) vs. Fitted)
plt.figure(figsize=(8,5))
plt.scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.6, edgecolor='k')
plt.xlabel('Fitted values')
plt.ylabel('Sqrt(|Standardized Residuals|)')
plt.title('Scale-Location Plot')
plt.tight_layout()
plt.savefig('level_1_visualizations/scale_location_plot.png')
plt.close()
