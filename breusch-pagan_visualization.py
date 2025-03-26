import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from statsmodels.stats.diagnostic import het_breuschpagan

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

# ------------------------------
# Heteroscedasticity Visualization using Breusch-Pagan Test
# ------------------------------
# For Good Model
resid_good = model_good.resid
fitted_good = model_good.fittedvalues
sq_resid_good = resid_good**2

# Breusch-Pagan test for Good Model
bp_test_good = het_breuschpagan(model_good.resid, model_good.model.exog)
bp_pvalue_good = bp_test_good[1]

# For Poor Model
resid_poor = model_poor.resid
fitted_poor = model_poor.fittedvalues
sq_resid_poor = resid_poor**2

# Breusch-Pagan test for Poor Model
bp_test_poor = het_breuschpagan(model_poor.resid, model_poor.model.exog)
bp_pvalue_poor = bp_test_poor[1]

# Create a combined plot for heteroscedasticity visualization
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Good Model: Squared Residuals vs Fitted Values
axs[0].scatter(fitted_good, sq_resid_good, alpha=0.7, label='Squared Residuals')
z_good = np.polyfit(fitted_good, sq_resid_good, 1)
p_good = np.poly1d(z_good)
axs[0].plot(fitted_good, p_good(fitted_good), "r--", label='Trend Line')
axs[0].set_xlabel('Fitted Values')
axs[0].set_ylabel('Squared Residuals')
axs[0].set_title(f'Good Model\nBP p-value: {bp_pvalue_good:.3f}')
axs[0].legend()

# Poor Model: Squared Residuals vs Fitted Values
axs[1].scatter(fitted_poor, sq_resid_poor, alpha=0.7, label='Squared Residuals')
z_poor = np.polyfit(fitted_poor, sq_resid_poor, 1)
p_poor = np.poly1d(z_poor)
axs[1].plot(fitted_poor, p_poor(fitted_poor), "r--", label='Trend Line')
axs[1].set_xlabel('Fitted Values')
axs[1].set_ylabel('Squared Residuals')
axs[1].set_title(f'Poor Model\nBP p-value: {bp_pvalue_poor:.3f}')
axs[1].legend()

fig.tight_layout()
fig.savefig('images/heteroscedasticity_bp_comparison.png')
plt.close(fig)
