import numpy as np
import pandas as pd
import os

# Ensure the 'data' subfolder exists
os.makedirs('data', exist_ok=True)

# For reproducibility
np.random.seed(0)
n = 2798

# Define five regions: A, B, C, D, E.
regions = ['A', 'B', 'C', 'D', 'E']
region = np.random.choice(regions, size=n)

# Generate predictors with updated ranges:
area = np.random.uniform(500, 1200000, size=n)               # 500 to 1,200,000 sqm
distance_capital = np.random.uniform(0.5, 750, size=n)         # 0.5 to 750 km
distance_elevator = np.random.uniform(0.5, 350, size=n)        # 0.5 to 350 km
access = np.random.binomial(1, 0.7, size=n)
coast_line = np.random.binomial(1, 0.3, size=n)
ownership = np.random.binomial(1, 0.6, size=n)
crop_yield = np.random.randint(35, 76, size=n)
simple_shape = np.random.binomial(1, 0.8, size=n)
is_marked = np.random.binomial(1, 0.5, size=n)
is_ab = np.array([1 if r in ['A','B'] else 0 for r in region])
is_cd = np.array([1 if r in ['C','D'] else 0 for r in region])
is_north_forest_steppe = np.random.binomial(1, 0.4, size=n)
is_south_forest_steppe = np.random.binomial(1, 0.3, size=n)
is_steppe = np.random.binomial(1, 0.5, size=n)

# True coefficients (log-scale), adjusted constant for higher price range
const = 4.5       # increases baseline price
beta_area = -0.2
beta_capital = -0.3
beta_elevator = -0.25
beta_yield = 0.3
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

# Construct log(price_per_sqm)
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

# Exponentiate to get price_per_sqm
price_per_sqm = np.exp(log_price_per_sqm)

# Clip to desired range 0.1 to 35 for realism
# price_per_sqm = np.clip(price_per_sqm, 0.1, 35)

# Create DataFrame
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

# Save to CSV
df_level1.to_csv('data/data_level_1.csv', index=False)
print("Data saved to 'data/data_level_1.csv'")
