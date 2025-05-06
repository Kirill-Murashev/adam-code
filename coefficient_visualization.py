import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
from scipy.stats import norm

# Ensure output directory exists
os.makedirs('level_1_visualizations', exist_ok=True)

# Load Level I data
df = pd.read_csv('data/data_level_1.csv')

# Log-transform for multiplicative model
df['log_price']             = np.log(df['price_per_sqm'])
df['log_area']              = np.log(df['area'])
df['log_distance_capital']  = np.log(df['distance_capital'])
df['log_distance_elevator'] = np.log(df['distance_elevator'])
df['log_crop_yield']        = np.log(df['crop_yield'])

# Fit the model
formula = (
    "log_price ~ "
    "log_area + log_distance_capital + log_distance_elevator + log_crop_yield + "
    "access + coast_line + ownership + simple_shape + is_marked + "
    "is_ab + is_cd + is_north_forest_steppe + is_south_forest_steppe + is_steppe"
)
model = smf.ols(formula, data=df).fit()

# Extract parameters and standard errors
params = model.params
std_errs = model.bse

# Prepare for plotting
predictors = params.index.tolist()
n = len(predictors)
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes = axes.flatten()

# Plot each coefficient distribution
for i, name in enumerate(predictors):
    mean = params[name]
    sd = std_errs[name]
    # Define x range
    x = np.linspace(mean - 4*sd, mean + 4*sd, 200)
    y = norm.pdf(x, loc=mean, scale=sd)
    ax = axes[i]
    # Density curve
    ax.plot(x, y, label='Density', color='black')
    # Mean line
    ax.axvline(mean, color='red', linestyle='-', label='Mean')
    # Standard deviation lines (±1 SD)
    ax.axvline(mean - sd, color='green', linestyle=':', label='Std Dev')
    ax.axvline(mean + sd, color='green', linestyle=':')
    # 95% CI lines
    z95 = norm.ppf(0.975)
    ci_lower = mean - z95*sd
    ci_upper = mean + z95*sd
    ax.axvline(ci_lower, color='blue', linestyle='--', label='95% CI Lower')
    ax.axvline(ci_upper, color='blue', linestyle='--', label='95% CI Upper')
    ax.set_title(name)
    ax.legend()

# Remove any empty subplots
for j in range(n, nrows*ncols):
    fig.delaxes(axes[j])

fig.tight_layout()
fig.savefig("level_1_visualizations/coeff_distributions.png")
plt.close(fig)

