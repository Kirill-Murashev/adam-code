import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1) Load Level V data
df = pd.read_csv('data/data_level_5.csv')

# 2) Prepare features and response
df['log_price']            = np.log(df['price_per_sqm'])
df['log_area']             = np.log(df['area'])
df['log_distance_capital'] = np.log(df['distance_capital'])
# formula string
formula = 'log_price ~ log_area + log_distance_capital'

# 3) LOOCV
loo = LeaveOneOut()
y_true, y_pred = [], []

for train_idx, test_idx in loo.split(df):
    df_train = df.iloc[train_idx]
    df_test  = df.iloc[test_idx]

    # fit with formula API
    model = smf.ols(formula, data=df_train).fit()

    # predict on the single-row test‐set
    y_hat = model.predict(df_test).iloc[0]

    y_true.append(df_test['log_price'].iloc[0])
    y_pred.append(y_hat)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 4) Compute metrics
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

metrics = {
    'LOOCV_MSE':  mse,
    'LOOCV_RMSE': rmse,
    'LOOCV_MAE':  mae,
    'LOOCV_R2':   r2
}

# 5) Per‐fold errors
errors_df = pd.DataFrame({
    'y_true':        y_true,
    'y_pred':        y_pred,
    'squared_error': (y_true - y_pred)**2,
    'absolute_error': np.abs(y_true - y_pred)
})

# 6) Save
out_dir = 'cv_results_level_5'
os.makedirs(out_dir, exist_ok=True)
errors_df.to_csv(f'{out_dir}/loocv_errors.csv', index=False)
pd.DataFrame([metrics]).to_csv(f'{out_dir}/loocv_metrics.csv', index=False)

# 7) Visualize
plt.figure(figsize=(6,4))
plt.boxplot([errors_df['squared_error'], errors_df['absolute_error']],
            tick_labels=['Squared Error','Absolute Error'])
plt.title('LOOCV Error Distributions (Level V)')
plt.ylabel('Error')
plt.tight_layout()
plt.savefig(f'{out_dir}/loocv_error_boxplots.png')
plt.close()

plt.figure(figsize=(6,4))
plt.hist(errors_df['absolute_error'], bins=10, density=True, alpha=0.7)
plt.title('LOOCV Absolute Error Density (Level V)')
plt.xlabel('Absolute Error')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(f'{out_dir}/loocv_abs_error_density.png')
plt.close()
