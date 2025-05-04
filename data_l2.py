import numpy as np
import pandas as pd
import os

# Ensure the 'data' subfolder exists
if not os.path.exists('data'):
    os.makedirs('data')

# For reproducibility
np.random.seed(1)
n = 714

# Generate predictors for Level II (target region)
# 1. area (in square meters): larger plots tend to have lower unit price
area = np.random.uniform(2000, 1100000, size=n)
# 2. distance from regional capital in km: closer is better
distance_capital = np.random.uniform(0.5, 395, size=n)
# 3. distance from elevator in km: closer is better
distance_elevator = np.random.uniform(0.5, 205, size=n)
# 4. access from paved road: binary, 1 with probability 0.7
access = np.random.binomial(1, 0.7, size=n)
# 5. ownership: binary, 1 (private) with probability 0.6
ownership = np.random.binomial(1, 0.6, size=n)
# 6. crop_yield: integer between 35 and 75
crop_yield = np.random.randint(40, 73, size=n)
# 7. simple_shape: binary, 1 with probability 0.8
simple_shape = np.random.binomial(1, 0.8, size=n)
# 8. is_marked: binary, 1 with probability 0.5
is_marked = np.random.binomial(1, 0.5, size=n)
# 9. is_north_forest_steppe: binary, probability 0.4
is_north_forest_steppe = np.random.binomial(1, 0.4, size=n)
# 10. is_south_forest_steppe: binary, probability 0.3
is_south_forest_steppe = np.random.binomial(1, 0.3, size=n)
# 11. is_steppe: binary, probability 0.5
is_steppe = np.random.binomial(1, 0.5, size=n)

# True coefficients for multiplicative model (log-scale)
const = 5.0
beta_area = -0.23
beta_capital = -0.37
beta_elevator = -0.21
beta_yield = 0.35
gamma_access = 0.12
gamma_ownership = 0.11
gamma_shape = 0.1
gamma_marked = 0.13
gamma_north = 0.06
gamma_south = 0.15
gamma_steppe = 0.09

# Generate error term ~ N(0, 0.2) for higher precision
error = np.random.normal(0, 0.2, n)

# Construct log(price_per_sqm)
log_price_per_sqm = (
    const +
    beta_area * np.log(area) +
    beta_capital * np.log(distance_capital) +
    beta_elevator * np.log(distance_elevator) +
    beta_yield * np.log(crop_yield) +
    gamma_access * access +
    gamma_ownership * ownership +
    gamma_shape * simple_shape +
    gamma_marked * is_marked +
    gamma_north * is_north_forest_steppe +
    gamma_south * is_south_forest_steppe +
    gamma_steppe * is_steppe +
    error
)

# Exponentiate to get price_per_sqm
price_per_sqm = np.exp(log_price_per_sqm)

# Create DataFrame
df_level2 = pd.DataFrame({
    'area': area,
    'distance_capital': distance_capital,
    'distance_elevator': distance_elevator,
    'access': access,
    'ownership': ownership,
    'crop_yield': crop_yield,
    'simple_shape': simple_shape,
    'is_marked': is_marked,
    'is_north_forest_steppe': is_north_forest_steppe,
    'is_south_forest_steppe': is_south_forest_steppe,
    'is_steppe': is_steppe,
    'price_per_sqm': price_per_sqm
})

# Save to CSV
df_level2.to_csv('data/data_level_2.csv', index=False)

