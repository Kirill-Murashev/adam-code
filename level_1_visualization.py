import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# Ensure the 'output_charts' subfolder exists
if not os.path.exists('output_charts'):
    os.makedirs('output_charts')

# 1) Read the data from CSV
df = pd.read_csv('data/data_level_1.csv')

# Create a log-transformed column for price_per_sqm
df['log_price'] = np.log(df['price_per_sqm'])

##############################################
# 2) Combined 2x2 Chart for Price Distribution
##############################################
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Top Left: Histogram (natural scale)
axs[0, 0].hist(df['price_per_sqm'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axs[0, 0].set_title("Histogram of Price per sqm (Natural Scale)")
axs[0, 0].set_xlabel("Price per sqm")
axs[0, 0].set_ylabel("Frequency")

# Top Right: Histogram (log scale) with overlaid normal curve
n, bins, patches = axs[0, 1].hist(df['log_price'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7,
                                  density=True)
# Fit a normal distribution to the log data
mu, std = stats.norm.fit(df['log_price'])
xmin, xmax = axs[0, 1].get_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
axs[0, 1].plot(x, p, 'r--', linewidth=2)
axs[0, 1].set_title("Histogram of log(Price per sqm) with Normal Curve")
axs[0, 1].set_xlabel("log(Price per sqm)")
axs[0, 1].set_ylabel("Density")

# Bottom Left: KDE plot (natural scale) with fitted normal curve
sns.kdeplot(df['price_per_sqm'], ax=axs[1, 0], color='blue', fill=True, label='KDE')
# Fit normal on natural scale
mu_nat, std_nat = stats.norm.fit(df['price_per_sqm'])
x_nat = np.linspace(df['price_per_sqm'].min(), df['price_per_sqm'].max(), 100)
p_nat = stats.norm.pdf(x_nat, mu_nat, std_nat)
axs[1, 0].plot(x_nat, p_nat, 'r--', linewidth=2, label='Normal Fit')
axs[1, 0].set_title("KDE of Price per sqm (Natural Scale)")
axs[1, 0].set_xlabel("Price per sqm")
axs[1, 0].set_ylabel("Density")
axs[1, 0].legend()

# Bottom Right: KDE plot (log scale) with fitted normal curve
sns.kdeplot(df['log_price'], ax=axs[1, 1], color='purple', fill=True, label='KDE')
axs[1, 1].plot(x, p, 'r--', linewidth=2, label='Normal Fit')
axs[1, 1].set_title("KDE of log(Price per sqm)")
axs[1, 1].set_xlabel("log(Price per sqm)")
axs[1, 1].set_ylabel("Density")
axs[1, 1].legend()

fig.tight_layout()
fig.savefig("output_charts/level_1_price_distribution.png")
plt.close(fig)

##################################################
# 3) Combined 2x2 Chart for Scatter Plots (log-log)
##################################################
# Create log-transformed versions for continuous predictors
df['log_area'] = np.log(df['area'])
df['log_distance_capital'] = np.log(df['distance_capital'])
df['log_distance_elevator'] = np.log(df['distance_elevator'])
df['log_crop_yield'] = np.log(df['crop_yield'])

scatter_vars = {
    'log_area': 'log(Area)',
    'log_distance_capital': 'log(Distance from Capital)',
    'log_distance_elevator': 'log(Distance from Elevator)',
    'log_crop_yield': 'log(Crop Yield)'
}

fig, axs = plt.subplots(2, 2, figsize=(14, 12))
axs = axs.flatten()

for ax, (var, label) in zip(axs, scatter_vars.items()):
    # Scatter plot of log_price vs log-transformed predictor
    ax.scatter(df[var], df['log_price'], alpha=0.5, color='teal')
    ax.set_xlabel(label)
    ax.set_ylabel("log(Price per sqm)")
    ax.set_title(f"log(Price) vs. {label}")

    # Calculate Pearson correlation coefficient and p-value
    r, p_value = stats.pearsonr(df[var], df['log_price'])
    ax.text(0.05, 0.95, f"r = {r:.2f}\np = {p_value:.3e}", transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))

fig.tight_layout()
fig.savefig("output_charts/level_1_scatterplots.png")
plt.close(fig)

#############################################################
# 4) Combined Violin Plots for Binary Predictors (log_price)
#############################################################
# List of binary predictors for which to create violin plots
binary_vars = ['access', 'coast_line', 'ownership', 'simple_shape', 'is_marked']

fig, axs = plt.subplots(1, len(binary_vars), figsize=(4*len(binary_vars), 6), sharey=True)
for ax, var in zip(axs, binary_vars):
    sns.violinplot(x=var, y='log_price', data=df, ax=ax, inner='quartile',
                   palette='pastel', scale='width', cut=0)
    ax.set_title(f"{var}", fontsize=12)
    # Compute Mann–Whitney U test between groups 0 and 1
    group0 = df[df[var] == 0]['log_price']
    group1 = df[df[var] == 1]['log_price']
    u_stat, p_val = stats.mannwhitneyu(group0, group1, alternative='two-sided')
    ax.text(0.5, 0.95, f"U = {u_stat:.1f}\np = {p_val:.3e}", transform=ax.transAxes,
            horizontalalignment='center', verticalalignment='top',
            bbox=dict(boxstyle="round", fc="w"), fontsize=10)
fig.suptitle("Violin Plots for Binary Predictors (log(Price per sqm))", y=0.98, fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("output_charts/level_1_violin_binary.png")
plt.close(fig)

##########################################################
# 5) Combined Violin Plots for Categorical Variables (Region & Zone)
##########################################################
# Create a new categorical variable for region grouping:
# Group as: AB (if region is A or B), CD (if region is C or D), and E.
df['region_group'] = df['region'].apply(lambda x: 'AB' if x in ['A', 'B']
else ('CD' if x in ['C', 'D'] else 'E'))


# Create a new categorical variable for zone.
# For simplicity, we'll assume the following rule:
# If is_north_forest_steppe==1 then "North"; else if is_south_forest_steppe==1 then "South";
# else if is_steppe==1 then "Steppe"; otherwise "Other".
def determine_zone(row):
    if row['is_north_forest_steppe'] == 1:
        return "North"
    elif row['is_south_forest_steppe'] == 1:
        return "South"
    elif row['is_steppe'] == 1:
        return "Steppe"
    else:
        return "Other"


df['zone'] = df.apply(determine_zone, axis=1)

# Violin plots for region_group and zone using log_price
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
sns.violinplot(x='region_group', y='log_price', data=df, ax=axs[0], palette='Set2',
               inner='quartile', scale='width', cut=0)
axs[0].set_title("log(Price per sqm) by Region Group", fontsize=12)

# ANOVA for region group
groups = [group['log_price'].values for name, group in df.groupby('region_group')]
F_stat, p_anova = stats.f_oneway(*groups)
axs[0].text(0.05, 0.95, f"F = {F_stat:.2f}\np = {p_anova:.3e}", transform=axs[0].transAxes,
            verticalalignment='top', bbox=dict(boxstyle="round", fc="w"), fontsize=10)

sns.violinplot(x='zone', y='log_price', data=df, ax=axs[1], palette='Set3',
               inner='quartile', scale='width', cut=0)
axs[1].set_title("log(Price per sqm) by Zone", fontsize=12)

# ANOVA for zone
groups_zone = [group['log_price'].values for name, group in df.groupby('zone')]
F_stat_zone, p_anova_zone = stats.f_oneway(*groups_zone)
axs[1].text(0.05, 0.95, f"F = {F_stat_zone:.2f}\np = {p_anova_zone:.3e}", transform=axs[1].transAxes,
            verticalalignment='top', bbox=dict(boxstyle="round", fc="w"), fontsize=10)

fig.suptitle("Violin Plots for Categorical Predictors (log(Price per sqm))", y=0.98, fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("output_charts/level_1_violin_categorical.png")
plt.close(fig)
