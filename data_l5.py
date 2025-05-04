import numpy as np
import pandas as pd
import os

# Ensure the 'data' subfolder exists
os.makedirs('data', exist_ok=True)

# For reproducibility
np.random.seed(5)
n = 27

# Generate predictors
area = np.random.uniform(5000, 55000, size=n)           # 5000 to 55,000 sqm
distance_capital = np.random.uniform(30, 80, size=n)  # 30 to 80 km
ownership = np.random.binomial(1, 0.6, size=n)          # private vs shared

# Additional (noise) variables
distance_elevator = np.random.uniform(0.5, 350, size=n)
access = np.random.binomial(1, 0.7, size=n)
coast_line = np.random.binomial(1, 0.3, size=n)
crop_yield = np.random.randint(65, 73, size=n)
simple_shape = np.random.binomial(1, 0.8, size=n)
is_marked = np.random.binomial(1, 0.5, size=n)

# True coefficients (log-scale)
const = 10.0
beta_area = -0.48
beta_capital = -0.59
gamma_ownership = 0.38

# Generate a mixed error: small normal on log-scale + additive noise on price-scale
log_error = np.random.normal(0, 0.1, size=n)
log_price = (const +
             beta_area * np.log(area) +
             beta_capital * np.log(distance_capital) +
             gamma_ownership * ownership +
             log_error)

# Base price on log-scale
price_base = np.exp(log_price)

# Additive noise to break pure log-normality
price_per_sqm = price_base + np.random.normal(0, price_base * 0.05, size=n)
price_per_sqm = np.maximum(price_per_sqm, 0.01)  # ensure positivity

# Assemble DataFrame
df_level5 = pd.DataFrame({
    'area': area,
    'distance_capital': distance_capital,
    'ownership': ownership,
    'distance_elevator': distance_elevator,
    'access': access,
    'coast_line': coast_line,
    'crop_yield': crop_yield,
    'simple_shape': simple_shape,
    'is_marked': is_marked,
    'price_per_sqm': price_per_sqm
})

# Save to CSV
df_level5.to_csv('data/data_level_5.csv', index=False)
print("Data saved to 'data/data_level_5.csv'")
