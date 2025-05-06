import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Ensure output directory exists
os.makedirs('level_1_visualizations', exist_ok=True)

# Load Level I data and fit model
df = pd.read_csv('data/data_level_1.csv')
df['log_price']             = np.log(df['price_per_sqm'])
df['log_area']              = np.log(df['area'])
df['log_distance_capital']  = np.log(df['distance_capital'])
df['log_distance_elevator'] = np.log(df['distance_elevator'])
df['log_crop_yield']        = np.log(df['crop_yield'])

formula = (
    "log_price ~ log_area + log_distance_capital + log_distance_elevator + "
    "log_crop_yield + access + coast_line + ownership + simple_shape + "
    "is_marked + is_ab + is_cd + is_north_forest_steppe + "
    "is_south_forest_steppe + is_steppe"
)
model = smf.ols(formula, data=df).fit()

# Extract residuals
resid = model.resid

# 1) ACF and PACF plots
fig, axes = plt.subplots(2, 1, figsize=(8, 10))
plot_acf(resid, lags=20, ax=axes[0])
axes[0].set_title('ACF of Residuals (Level I)')
plot_pacf(resid, lags=20, ax=axes[1])
axes[1].set_title('PACF of Residuals (Level I)')
plt.tight_layout()
plt.savefig('level_1_visualizations/residuals_acf_pacf.png')
plt.close(fig)

# 2) Ljung-Box p-values vs. lag
lb_results = acorr_ljungbox(resid, lags=np.arange(1, 21), return_df=True)
lags = lb_results.index
p_values = lb_results['lb_pvalue']

plt.figure(figsize=(8, 5))
plt.plot(lags, p_values, marker='o', linestyle='-')
plt.axhline(0.05, color='red', linestyle='--', label='0.05 Significance')
plt.xlabel('Lag')
plt.ylabel('Ljung–Box p-value')
plt.title('Ljung–Box p-values vs. Lag (Level I)')
plt.legend()
plt.tight_layout()
plt.savefig('level_1_visualizations/ljung_box_pvalues.png')
plt.close()

