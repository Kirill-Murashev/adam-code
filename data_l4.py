import numpy as np
import pandas as pd
import os

# Ensure the 'data' subfolder exists
if not os.path.exists('data'):
    os.makedirs('data')

# For reproducibility
np.random.seed(3)
n = 255

# Generate predictors for Level III (southern climate zone)
area = np.random.uniform(3500, 350000, size=n)
distance_capital = np.random.uniform(0.5, 110, size=n)
distance_elevator = np.random.uniform(0.5, 80, size=n)
access = np.random.binomial(1, 0.7, size=n)
ownership = np.random.binomial(1, 0.6, size=n)
crop_yield = np.random.randint(60, 73, size=n)
simple_shape = np.random.binomial(1, 0.7, size=n)
is_marked = np.random.binomial(1, 0.5, size=n)

# Southern climate zone subzones: southern forest-steppe vs steppe
# All observations in southern zone: no northern forest-steppe
is_north_forest_steppe = np.zeros(n, dtype=int)
# Assign subzones: 60% southern forest-steppe (more expensive), 40% steppe
is_south_forest_steppe = np.random.binomial(1, 0.6, size=n)
is_steppe = 1 - is_south_forest_steppe

# True coefficients for multiplicative model (log-scale), increased mean price and influence
const = 9.5
beta_area = -0.45
beta_capital = -0.55
beta_elevator = -0.28
gamma_ownership = 0.30
gamma_shape = 0.22


# Reduced noise for higher precision in local zone
error = np.random.normal(0, 0.15, n)

# Construct log(price_per_sqm)
log_price_per_sqm = (
    const +
    beta_area * np.log(area) +
    beta_capital * np.log(distance_capital) +
    beta_elevator * np.log(distance_elevator) +
    gamma_ownership * ownership +
    gamma_shape * simple_shape +
    error
)

# Exponentiate to get price_per_sqm
price_per_sqm = np.exp(log_price_per_sqm)

# Create DataFrame
df_level3 = pd.DataFrame({
    'area': area,
    'distance_capital': distance_capital,
    'distance_elevator': distance_elevator,
    'access': access,
    'ownership': ownership,
    'crop_yield': crop_yield,
    'simple_shape': simple_shape,
    'is_marked': is_marked,
    'is_south_forest_steppe': is_south_forest_steppe,
    'price_per_sqm': price_per_sqm
})

# Save to CSV
df_level3.to_csv('data/data_level_4.csv', index=False)
print("Data saved to 'data/data_level_4.csv'")
