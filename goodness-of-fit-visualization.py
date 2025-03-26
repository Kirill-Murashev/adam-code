import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# For reproducibility
np.random.seed(0)
n = 500

# Generate predictors common to both datasets:
# Two continuous variables: X1 and X2
X1 = np.random.uniform(50, 200, size=n)  # e.g., area
X2 = np.random.uniform(10, 100, size=n)  # e.g., distance from capital
# Two binary variables: X3 and X4 (e.g., presence of electricity, and another binary feature)
X3 = np.random.binomial(1, 0.5, size=n)
X4 = np.random.binomial(1, 0.5, size=n)

# ------------------------------
# Generate Good Model Data
# ------------------------------
# Coefficients chosen to yield a high R2 (small error variance)
beta0_good = 2.0
beta1_good = 0.5  # elasticity for log(X1)
beta2_good = -0.3  # elasticity for log(X2)
beta3_good = 0.2  # effect for binary X3
beta4_good = 0.5  # effect for binary X4

# Small error variance for high performance
error_good = np.random.normal(0, 0.2, size=n)
# Construct log(Y) using log-transformed continuous predictors and binary predictors as-is.
logY_good = beta0_good + beta1_good * np.log(X1) + beta2_good * np.log(
    X2) + beta3_good * X3 + beta4_good * X4 + error_good
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
error_poor = np.random.normal(0, 1.0, size=n)
logY_poor = beta0_poor + beta1_poor * np.log(X1) + beta2_poor * np.log(
    X2) + beta3_poor * X3 + beta4_poor * X4 + error_poor
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


# ------------------------------
# Diagnostic Plots Function
# ------------------------------

def diagnostic_plots(model, df, model_name):
    # Extract fitted values and residuals
    fitted_vals = model.fittedvalues
    residuals = model.resid
    # Influence measures for Cook's Distance and leverage
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag
    std_resid = influence.resid_studentized_internal

    # Create a figure with 2 rows x 3 columns of subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    axs = axs.flatten()

    # 1. Residuals vs Fitted
    axs[0].scatter(fitted_vals, residuals, alpha=0.7)
    axs[0].axhline(0, color='red', linestyle='--')
    axs[0].set_title('Residuals vs Fitted')
    axs[0].set_xlabel('Fitted values')
    axs[0].set_ylabel('Residuals')

    # 2. Observed vs Predicted Values
    # Convert predicted log(Y) back to Y
    predicted_Y = np.exp(fitted_vals)
    axs[1].scatter(predicted_Y, df['Y'], alpha=0.7)
    axs[1].plot([df['Y'].min(), df['Y'].max()], [df['Y'].min(), df['Y'].max()], 'r--')
    axs[1].set_title('Observed vs Predicted')
    axs[1].set_xlabel('Predicted Y')
    axs[1].set_ylabel('Observed Y')

    # 3. Q-Q Plot of Residuals
    sm.qqplot(residuals, line='45', ax=axs[2])
    axs[2].set_title('Q-Q Plot')

    # 4. Histogram of Residuals
    axs[3].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axs[3].set_title('Residual Histogram')
    axs[3].set_xlabel('Residuals')
    axs[3].set_ylabel('Frequency')

    fig.suptitle(f'Diagnostic Plots for {model_name}', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# Generate diagnostic plots for both models (assuming model_good, df_good, model_poor, and df_poor are already defined)
fig_good = diagnostic_plots(model_good, df_good, 'Good Model')
fig_poor = diagnostic_plots(model_poor, df_poor, 'Poor Model')

# Ensure the images subfolder exists
if not os.path.exists('images'):
    os.makedirs('images')

# Save the figures to separate files
fig_good.savefig('images/diagnostic_good_model.png')
fig_poor.savefig('images/diagnostic_poor_model.png')

plt.show()

