import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from statsmodels.stats.diagnostic import acorr_ljungbox

# For reproducibility
np.random.seed(0)
n = 500

# Generate predictors common to both datasets:
# Two continuous variables: X1 and X2
X1 = np.random.uniform(50, 200, size=n)  # e.g., area
X2 = np.random.uniform(10, 100, size=n)   # e.g., distance from capital
# Two binary variables: X3 and X4 (e.g., presence of electricity, and another binary feature)
X3 = np.random.binomial(1, 0.5, size=n)
X4 = np.random.binomial(1, 0.5, size=n)

# ------------------------------
# Generate Good Model Data
# ------------------------------
# Coefficients chosen to yield a high R2 (small error variance)
beta0_good = 2.0
beta1_good = 0.5    # elasticity for log(X1)
beta2_good = -0.3   # elasticity for log(X2)
beta3_good = 0.2    # effect for binary X3
beta4_good = 0.5    # effect for binary X4

# Small error variance for high performance
error_good = np.random.normal(0, 0.2, size=n)
# Construct log(Y) using log-transformed continuous predictors and binary predictors as-is.
logY_good = beta0_good + beta1_good * np.log(X1) + beta2_good * np.log(X2) + beta3_good * X3 + beta4_good * X4 + error_good
Y_good = np.exp(logY_good)  # Y is log-normally distributed
df_good = pd.DataFrame({
    'Y': Y_good,
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'X4': X4
})

# ------------------------------
# Generate Poor Model Data with Heteroscedasticity
# ------------------------------
# Coefficients chosen so that predictors explain little of the variance.
beta0_poor = 2.0
beta1_poor = 0.1
beta2_poor = 0.1
beta3_poor = 0.0
beta4_poor = 0.0

# Create heteroscedastic errors: standard deviation increases with X1.
error_poor = np.array([np.random.normal(0, 1.0 + 0.01 * x) for x in X1])
logY_poor = beta0_poor + beta1_poor * np.log(X1) + beta2_poor * np.log(X2) + beta3_poor * X3 + beta4_poor * X4 + error_poor
Y_poor = np.exp(logY_poor)
df_poor = pd.DataFrame({
    'Y': Y_poor,
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'X4': X4
})

# ------------------------------
# Prepare Data for Regression
# ------------------------------
# We model log(Y) as a function of log(X1), log(X2) and the binary variables X3 and X4.
for df in [df_good, df_poor]:
    df['logY'] = np.log(df['Y'])
    df['logX1'] = np.log(df['X1'])
    df['logX2'] = np.log(df['X2'])

# Add constant for intercept and fit models
X_good = sm.add_constant(df_good[['logX1', 'logX2', 'X3', 'X4']])
model_good = sm.OLS(df_good['logY'], X_good).fit()

X_poor = sm.add_constant(df_poor[['logX1', 'logX2', 'X3', 'X4']])
model_poor = sm.OLS(df_poor['logY'], X_poor).fit()

print("Good Model Summary:")
print(model_good.summary())
print("\nPoor Model Summary:")
print(model_poor.summary())

# Ensure the 'images' subfolder exists
if not os.path.exists('images'):
    os.makedirs('images')

# 1. ACF and PACF plots for the residuals (using the Good Model)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sm.graphics.tsa.plot_acf(model_good.resid, lags=20, ax=axes[0])
axes[0].set_title('ACF of Residuals (Good Model)')
sm.graphics.tsa.plot_pacf(model_good.resid, lags=20, ax=axes[1])
axes[1].set_title('PACF of Residuals (Good Model)')
fig.tight_layout()
fig.savefig('images/ACF_PACF_Good_Model.png')
plt.close(fig)

# 2. Residuals vs. Lag Plot for the Good Model
fig, ax = plt.subplots(figsize=(10, 5))
res_good = model_good.resid.values  # Convert to numpy array
lags = np.arange(1, len(res_good))
diffs_good = res_good[1:] - res_good[:-1]
ax.plot(lags, diffs_good, marker='o', linestyle='-', color='blue')
ax.axhline(0, color='red', linestyle='--')
ax.set_title('Residual Differences vs. Lag (Good Model)')
ax.set_xlabel('Lag')
ax.set_ylabel('Difference of Residuals')
fig.tight_layout()
fig.savefig('images/Residuals_vs_Lag_Good_Model.png')
plt.close(fig)

# 3. Ljung-Box test p-values across multiple lags for the Good Model
# Compute Ljung-Box test p-values for lags 1 to 20
ljung_box_results = acorr_ljungbox(model_good.resid, lags=np.arange(1, 21), return_df=True)
lags = ljung_box_results.index
p_values = ljung_box_results['lb_pvalue']

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(lags, p_values, marker='s', linestyle='-', color='green')
ax.axhline(0.05, color='red', linestyle='--', label='Significance Level (0.05)')
ax.set_title('Ljung-Box Test p-values vs. Lag (Good Model)')
ax.set_xlabel('Lag')
ax.set_ylabel('p-value')
ax.legend()
fig.tight_layout()
fig.savefig('images/Ljung_Box_pvalues_Good_Model.png')
plt.close(fig)

# 4. ACF and PACF plots for the residuals (using the Poor Model)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sm.graphics.tsa.plot_acf(model_poor.resid, lags=20, ax=axes[0])
axes[0].set_title('ACF of Residuals (Poor Model)')
sm.graphics.tsa.plot_pacf(model_poor.resid, lags=20, ax=axes[1])
axes[1].set_title('PACF of Residuals (Poor Model)')
fig.tight_layout()
fig.savefig('images/ACF_PACF_Poor_Model.png')
plt.close(fig)

# 5. Residuals vs. Lag Plot for the Poor Model
fig, ax = plt.subplots(figsize=(10, 5))
res_poor = model_poor.resid.values  # Convert to numpy array
lags = np.arange(1, len(res_poor))
diffs_poor = res_poor[1:] - res_poor[:-1]
ax.plot(lags, diffs_poor, marker='o', linestyle='-', color='blue')
ax.axhline(0, color='red', linestyle='--')
ax.set_title('Residual Differences vs. Lag (Poor Model)')
ax.set_xlabel('Lag')
ax.set_ylabel('Difference of Residuals')
fig.tight_layout()
fig.savefig('images/Residuals_vs_Lag_Poor_Model.png')
plt.close(fig)

# 6. Ljung-Box test p-values across multiple lags for the Poor Model
# Compute Ljung-Box test p-values for lags 1 to 20
ljung_box_results = acorr_ljungbox(model_poor.resid, lags=np.arange(1, 21), return_df=True)
lags = ljung_box_results.index
p_values = ljung_box_results['lb_pvalue']

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(lags, p_values, marker='s', linestyle='-', color='green')
ax.axhline(0.05, color='red', linestyle='--', label='Significance Level (0.05)')
ax.set_title('Ljung-Box Test p-values vs. Lag (Poor Model)')
ax.set_xlabel('Lag')
ax.set_ylabel('p-value')
ax.legend()
fig.tight_layout()
fig.savefig('images/Ljung_Box_pvalues_Poor_Model.png')
plt.close(fig)
