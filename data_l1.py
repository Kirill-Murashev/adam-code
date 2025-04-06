import numpy as np
import pandas as pd
import os

# Ensure the 'data' subfolder exists
if not os.path.exists('data'):
    os.makedirs('data')

# For reproducibility
np.random.seed(0)
n = 2798

# Define five regions: A, B, C, D, E.
regions = ['A', 'B', 'C', 'D', 'E']

# Generate region for each observation
region = np.random.choice(regions, size=n)

# Generate predictors:
# 1.1. area (in square meters): larger plots tend to have lower unit price
area = np.random.uniform(500, 10000, size=n)
# 1.2. distance from regional capital in km: closer is better (lower distance yields higher price)
distance_capital = np.random.uniform(0.5, 100, size=n)
# 1.3. distance from elevator in km: closer is better
distance_elevator = np.random.uniform(0.2, 50, size=n)
# 1.4. access from paved road: binary, 1 with probability 0.7
access = np.random.binomial(1, 0.7, size=n)
# 1.5. coast_line: binary, 1 with probability 0.3
coast_line = np.random.binomial(1, 0.3, size=n)
# 1.6. ownership: binary, 1 (private) with probability 0.6
ownership = np.random.binomial(1, 0.6, size=n)
# 1.7. crop_yield: integer between 35 and 75
crop_yield = np.random.randint(35, 76, size=n)
# 1.8. simple_shape: binary, 1 with probability 0.8
simple_shape = np.random.binomial(1, 0.8, size=n)
# 1.9. is_marked: binary, 1 with probability 0.5
is_marked = np.random.binomial(1, 0.5, size=n)
# 1.10. is_ab: 1 if region is A or B, else 0
is_ab = np.array([1 if r in ['A', 'B'] else 0 for r in region])
# 1.11. is_cd: 1 if region is C or D, else 0
is_cd = np.array([1 if r in ['C', 'D'] else 0 for r in region])
# 1.12. is_north_forest_steppe: binary, probability 0.4
is_north_forest_steppe = np.random.binomial(1, 0.4, size=n)
# 1.13. is_south_forest_steppe: binary, probability 0.3
is_south_forest_steppe = np.random.binomial(1, 0.3, size=n)
# 1.14. is_steppe: binary, probability 0.5
is_steppe = np.random.binomial(1, 0.5, size=n)

# Define true coefficients for the multiplicative model (on the log-scale)
const = 2.0
beta_area = -0.2
beta_capital = -0.3
beta_elevator = -0.25
beta_yield = 0.3  # Increased from a lower value to ensure statistical significance
gamma_access = 0.1
gamma_coast = 0.15
gamma_ownership = 0.12
gamma_shape = 0.08
gamma_marked = 0.1
gamma_ab = 0.2
gamma_cd = 0.1
gamma_north = 0.05
gamma_south = 0.12
gamma_steppe = 0.07

# Generate error term ~ N(0, 0.3)
error = np.random.normal(0, 0.3, n)

# Construct log(price_per_sqm) using the multiplicative model:
log_price_per_sqm = (
    const +
    beta_area * np.log(area) +
    beta_capital * np.log(distance_capital) +
    beta_elevator * np.log(distance_elevator) +
    beta_yield * np.log(crop_yield) +
    gamma_access * access +
    gamma_coast * coast_line +
    gamma_ownership * ownership +
    gamma_shape * simple_shape +
    gamma_marked * is_marked +
    gamma_ab * is_ab +
    gamma_cd * is_cd +
    gamma_north * is_north_forest_steppe +
    gamma_south * is_south_forest_steppe +
    gamma_steppe * is_steppe +
    error
)

# Exponentiate to obtain price_per_sqm; this should produce a log-normal distribution.
price_per_sqm = np.exp(log_price_per_sqm)

# Create a DataFrame
df_level1 = pd.DataFrame({
    'region': region,
    'area': area,
    'distance_capital': distance_capital,
    'distance_elevator': distance_elevator,
    'access': access,
    'coast_line': coast_line,
    'ownership': ownership,
    'crop_yield': crop_yield,
    'simple_shape': simple_shape,
    'is_marked': is_marked,
    'is_ab': is_ab,
    'is_cd': is_cd,
    'is_north_forest_steppe': is_north_forest_steppe,
    'is_south_forest_steppe': is_south_forest_steppe,
    'is_steppe': is_steppe,
    'price_per_sqm': price_per_sqm
})

# Save the generated data to CSV
df_level1.to_csv('data/data_level_1.csv', index=False)
print("Data saved to 'data/data_level_1.csv'")
