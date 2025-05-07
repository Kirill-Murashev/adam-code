import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence

# Ensure output directory exists
os.makedirs('level_1_visualizations', exist_ok=True)

# Load Level I data and fit the model
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

# Compute influence measures
infl = OLSInfluence(model)
cooks_d = infl.cooks_distance[0]
leverage = infl.hat_matrix_diag
dffits = infl.dffits[0]
dfbetas = infl.dfbetas  # shape (n_obs, n_params)

n = len(df)
p = len(model.params)

# Thresholds
cook_thresh = 4 / n
lev_thresh = 2 * p / n
dffits_thresh = 2 * np.sqrt(p / n)
dfbetas_thresh = 2 / np.sqrt(n)

# Plot all four diagnostics in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1) Cook's Distance
axes[0, 0].stem(np.arange(n), cooks_d, basefmt=" ")
axes[0, 0].axhline(cook_thresh, color='red', linestyle='--', label=f'4/n = {cook_thresh:.4f}')
axes[0, 0].set_title("Cook's Distance")
axes[0, 0].set_xlabel("Observation")
axes[0, 0].set_ylabel("Cook's D")
axes[0, 0].legend()

# 2) Leverage
axes[0, 1].scatter(np.arange(n), leverage, edgecolor='k', alpha=0.7)
axes[0, 1].axhline(lev_thresh, color='red', linestyle='--', label=f'2p/n = {lev_thresh:.4f}')
axes[0, 1].set_title("Leverage (Hat Values)")
axes[0, 1].set_xlabel("Observation")
axes[0, 1].set_ylabel("Leverage")
axes[0, 1].legend()

# 3) DFFITS
axes[1, 0].stem(np.arange(n), dffits, basefmt=" ")
axes[1, 0].axhline(dffits_thresh, color='red', linestyle='--', label=f'2√(p/n) = {dffits_thresh:.4f}')
axes[1, 0].axhline(-dffits_thresh, color='red', linestyle='--')
axes[1, 0].set_title("DFFITS")
axes[1, 0].set_xlabel("Observation")
axes[1, 0].set_ylabel("DFFITS")
axes[1, 0].legend()

# 4) DFBETAS - Boxplot for each coefficient
dfbetas_df = pd.DataFrame(dfbetas, columns=model.params.index)
# Fixed the deprecated 'labels' parameter to 'tick_labels'
axes[1, 1].boxplot([dfbetas_df[col] for col in dfbetas_df.columns],
                  tick_labels=dfbetas_df.columns, vert=False)
axes[1, 1].axvline(dfbetas_thresh, color='red', linestyle='--', label=f'±2/√n = {dfbetas_thresh:.4f}')
axes[1, 1].axvline(-dfbetas_thresh, color='red', linestyle='--')
axes[1, 1].set_title("DFBETAS (per coefficient)")
axes[1, 1].set_xlabel("DFBETAS")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("level_1_visualizations/influence_leverage_df.png")
plt.close(fig)