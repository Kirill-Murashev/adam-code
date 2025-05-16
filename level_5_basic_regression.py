import pandas as pd
import numpy as np
from primary_models import regression_diagnostics

# 1. Load Level V data
df_level1 = pd.read_csv('data/data_level_5.csv')

# 2. Create log‑transformed columns for the multiplicative model
df_level1['log_price']            = np.log(df_level1['price_per_sqm'])
df_level1['log_area']             = np.log(df_level1['area'])
df_level1['log_distance_capital'] = np.log(df_level1['distance_capital'])

# 3. Specify your formula (add any additional predictors if desired)
formula = (
    "log_price ~ "
    "log_area + log_distance_capital"
)

# 4. Run diagnostics
metrics_df, influence_df = regression_diagnostics(
    df=df_level1,
    formula=formula,
    out_prefix="level5"
)

# 5. (Optional) inspect key metrics in the console
print(metrics_df)
