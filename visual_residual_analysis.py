import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm

# For reproducibility
np.random.seed(0)
n = 500

# Generate predictors common to both datasets:
# Two continuous variables: X1 and X2
X1 = np.random.uniform(50, 200, size=n)   # e.g., area
X2 = np.random.uniform(10, 100, size=n)    # e.g., distance from capital
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
# Generate Poor Model Data
# ------------------------------
# Coefficients chosen so that predictors explain little of the variance,
# and with high error variance to yield a low R2.
beta0_poor = 2.0
beta1_poor = 0.1
beta2_poor = 0.1
beta3_poor = 0.0
beta4_poor = 0.0

# Large error variance to degrade model performance
error_poor = np.random.normal(0, 0.1, size=n)
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

# Add constant for intercept
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

# 1) Combined QQ-Plots for Good and Poor Models
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
sm.qqplot(model_good.resid, line='45', ax=axs[0])
axs[0].set_title('QQ-Plot: Good Model Residuals')
sm.qqplot(model_poor.resid, line='45', ax=axs[1])
axs[1].set_title('QQ-Plot: Poor Model Residuals')
plt.tight_layout()
fig.savefig('images/combined_QQ_plots.png')
plt.close(fig)

# 2) Combined Histograms for Good and Poor Models
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].hist(model_good.resid, bins=20, edgecolor='black', alpha=0.7)
axs[0].set_title('Histogram: Good Model Residuals')
axs[0].set_xlabel('Residuals')
axs[0].set_ylabel('Frequency')
axs[1].hist(model_poor.resid, bins=20, edgecolor='black', alpha=0.7)
axs[1].set_title('Histogram: Poor Model Residuals')
axs[1].set_xlabel('Residuals')
axs[1].set_ylabel('Frequency')
plt.tight_layout()
fig.savefig('images/combined_histograms.png')
plt.close(fig)

# 3) Combined Residuals vs Fitted Plots for Good and Poor Models
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].scatter(model_good.fittedvalues, model_good.resid, alpha=0.7)
axs[0].axhline(0, color='red', linestyle='--')
axs[0].set_title('Residuals vs Fitted: Good Model')
axs[0].set_xlabel('Fitted Values')
axs[0].set_ylabel('Residuals')
axs[1].scatter(model_poor.fittedvalues, model_poor.resid, alpha=0.7)
axs[1].axhline(0, color='red', linestyle='--')
axs[1].set_title('Residuals vs Fitted: Poor Model')
axs[1].set_xlabel('Fitted Values')
axs[1].set_ylabel('Residuals')
plt.tight_layout()
fig.savefig('images/combined_resid_vs_fitted.png')
plt.close(fig)
