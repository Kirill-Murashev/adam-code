import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns
from scipy.stats import norm

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
    "log_price ~ "
    "log_area + log_distance_capital + log_distance_elevator + log_crop_yield + "
    "access + coast_line + ownership + simple_shape + is_marked + "
    "is_ab + is_cd + is_north_forest_steppe + is_south_forest_steppe + is_steppe"
)
model = smf.ols(formula, data=df).fit()

# Extract residuals and fitted values
fitted = model.fittedvalues
resid = model.resid

# 1. Residual vs. Fitted with LOWESS
plt.figure(figsize=(8,5))
plt.scatter(fitted, resid, alpha=0.5, edgecolor='k')
smoothed = lowess(resid, fitted, frac=0.3)
plt.plot(smoothed[:,0], smoothed[:,1], 'r-', linewidth=2)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted (with LOWESS)')
plt.tight_layout()
plt.savefig('level_1_visualizations/resid_vs_fitted.png')
plt.close()

# 2. Q-Q Plot
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
qqplot(resid, line='45', ax=ax)
ax.set_title('Normal Q-Q Plot of Residuals')
plt.tight_layout()
plt.savefig('level_1_visualizations/qq_plot_residuals.png')
plt.close()

# 3. Histogram + KDE of Residuals
plt.figure(figsize=(8,5))
sns.histplot(resid, bins=30, kde=False, stat='density', color='skyblue', edgecolor='black', label='Histogram')
# KDE
sns.kdeplot(resid, color='navy', linewidth=2, label='KDE')
# Normal overlay
mu, std = norm.fit(resid)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r--', linewidth=2, label='Normal Fit')
plt.xlabel('Residual')
plt.ylabel('Density')
plt.title('Residual Distribution')
plt.legend()
plt.tight_layout()
plt.savefig('level_1_visualizations/resid_hist_kde.png')
plt.close()
