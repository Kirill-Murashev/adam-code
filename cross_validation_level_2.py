import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import os

# Ensure output directory exists
os.makedirs('mc_cv_l2', exist_ok=True)

# Load and prepare data
df = pd.read_csv('data/data_level_2.csv')
df['log_price'] = np.log(df['price_per_sqm'])
df['log_area'] = np.log(df['area'])
df['log_distance_capital'] = np.log(df['distance_capital'])
df['log_distance_elevator'] = np.log(df['distance_elevator'])
df['log_crop_yield'] = np.log(df['crop_yield'])

formula = (
    "log_price ~ "
    "log_area + log_distance_capital + log_distance_elevator + log_crop_yield + "
    "access + ownership + simple_shape + is_marked + is_north_forest_steppe + "
    "is_south_forest_steppe + is_steppe"
)

# Monte Carlo CV settings
n_splits = 10000
test_size = 0.2
rng = np.random.RandomState(0)

metrics_train = []
metrics_test = []

for i in range(n_splits):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=rng.randint(1e6))

    # Fit model on train
    model = smf.ols(formula, data=train_df).fit()

    # Train metrics
    y_train, y_train_pred = train_df['log_price'], model.fittedvalues
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = model.rsquared
    metrics_train.append({'MSE': train_mse, 'RMSE': train_rmse, 'MAE': train_mae, 'R2': train_r2})

    # Test metrics
    y_test, y_test_pred = test_df['log_price'], model.predict(test_df)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    metrics_test.append({'MSE': test_mse, 'RMSE': test_rmse, 'MAE': test_mae, 'R2': test_r2})

df_train = pd.DataFrame(metrics_train)
df_test = pd.DataFrame(metrics_test)

# Summary statistics
summary = pd.concat([df_train.describe(percentiles=[.1, .9]).add_prefix('train_'),
                     df_test.describe(percentiles=[.1, .9]).add_prefix('test_')], axis=1)
summary.to_csv('mc_cv_l2/summary_metrics.csv')

# Boxplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, metric in zip(axes.flatten(), ['R2', 'RMSE', 'MAE', 'MSE']):
    data = pd.DataFrame({'Train': df_train[metric], 'Test': df_test[metric]})
    ax.boxplot([data['Train'], data['Test']], tick_labels=['Train', 'Test'])
    ax.set_title(f'Boxplot of {metric} (Train vs Test)')
fig.tight_layout()
plt.savefig('mc_cv_l2/boxplots_metrics.png')
plt.close(fig)

# Density plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, metric in zip(axes.flatten(), ['R2', 'RMSE', 'MAE', 'MSE']):
    ax.hist(df_train[metric], bins=30, alpha=0.5, density=True, label='Train')
    ax.hist(df_test[metric], bins=30, alpha=0.5, density=True, label='Test')
    ax.set_title(f'Density of {metric}')
    ax.legend()
fig.tight_layout()
plt.savefig('mc_cv_l2/density_metrics.png')
plt.close(fig)

# Scatter train vs test R2
plt.figure(figsize=(6, 6))
plt.scatter(df_train['R2'], df_test['R2'], alpha=0.6)
plt.plot([df_train['R2'].min(), df_train['R2'].max()],
         [df_train['R2'].min(), df_train['R2'].max()],
         'r--', label='Ideal')
plt.xlabel('Train R²')
plt.ylabel('Test R²')
plt.title('Train vs Test R² per Split')
plt.legend()
plt.tight_layout()
plt.savefig('mc_cv_l2/scatter_r2.png')
plt.close()

# Save metrics data
df_train.to_csv('mc_cv_l2/train_metrics.csv', index=False)
df_test.to_csv('mc_cv_l2/test_metrics.csv', index=False)
summary
