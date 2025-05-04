import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Ensure output directory exists
os.makedirs('output_charts', exist_ok=True)

# Read Level I data
df = pd.read_csv('data/data_level_1.csv')
df['price_per_sqm'] = pd.to_numeric(df['price_per_sqm'], errors='coerce')
df['log_price'] = np.log(df['price_per_sqm'])

# 1) Price distribution (2×2)
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Top-left: natural histogram
axs[0, 0].hist(df['price_per_sqm'].dropna(), bins=50,
               color='skyblue', edgecolor='black', alpha=0.7)
axs[0, 0].set(title="Histogram of Price per sqm\n(Natural Scale)",
              xlabel="Price per sqm", ylabel="Frequency")

# Top-right: log-histogram + normal fit
axs[0, 1].hist(df['log_price'].dropna(), bins=50, density=True,
               color='lightgreen', edgecolor='black', alpha=0.7)
mu, std = stats.norm.fit(df['log_price'].dropna())
x = np.linspace(df['log_price'].min(), df['log_price'].max(), 100)
axs[0, 1].plot(x, stats.norm.pdf(x, mu, std), 'r--', lw=2)
axs[0, 1].set(title="Histogram of log(Price per sqm)\nwith Normal Fit",
              xlabel="log(Price per sqm)", ylabel="Density")

# Bottom-left: KDE (natural) + normal fit
kde_nat = df['price_per_sqm'].dropna().values.astype(float)
sns.kdeplot(x=kde_nat, ax=axs[1, 0], fill=True, color='blue', label='KDE')
mu_nat, std_nat = stats.norm.fit(kde_nat)
x_nat = np.linspace(kde_nat.min(), kde_nat.max(), 100)
axs[1, 0].plot(x_nat, stats.norm.pdf(x_nat, mu_nat, std_nat),
               'r--', lw=2, label='Normal Fit')
axs[1, 0].set(title="KDE of Price per sqm\n(Natural Scale)",
              xlabel="Price per sqm", ylabel="Density")
axs[1, 0].legend()

# Bottom-right: KDE (log) + normal fit
kde_log = df['log_price'].dropna().values.astype(float)
sns.kdeplot(x=kde_log, ax=axs[1, 1], fill=True, color='purple', label='KDE')
axs[1, 1].plot(x, stats.norm.pdf(x, mu, std), 'r--', lw=2, label='Normal Fit')
axs[1, 1].set(title="KDE of log(Price per sqm)",
              xlabel="log(Price per sqm)", ylabel="Density")
axs[1, 1].legend()

fig.tight_layout()
fig.savefig("output_charts/level_1_price_distribution.png")
plt.close(fig)

# 2) Scatterplots (2×2, log–log)
df['log_area']              = np.log(df['area'])
df['log_distance_capital']  = np.log(df['distance_capital'])
df['log_distance_elevator'] = np.log(df['distance_elevator'])
df['log_crop_yield']        = np.log(df['crop_yield'])

scatter_vars = {
    'log_area': 'log(Area)',
    'log_distance_capital': 'log(Distance from Capital)',
    'log_distance_elevator': 'log(Distance from Elevator)',
    'log_crop_yield': 'log(Crop Yield)'
}

fig, axs = plt.subplots(2, 2, figsize=(14, 12))
for ax, (var, label) in zip(axs.flatten(), scatter_vars.items()):
    x_vals = df[var].dropna().values
    y_vals = df['log_price'].dropna().values
    ax.scatter(x_vals, y_vals, alpha=0.5, color='teal')
    r, p = stats.pearsonr(x_vals, y_vals)
    ax.set(title=f"log(Price) vs. {label}", xlabel=label, ylabel="log(Price per sqm)")
    ax.text(0.05, 0.95, f"r = {r:.2f}\np = {p:.2e}",
            transform=ax.transAxes, va='top', bbox=dict(boxstyle="round", fc="w"))
fig.tight_layout()
fig.savefig("output_charts/level_1_scatterplots.png")
plt.close(fig)

# 3) Violin plots for binary predictors
binary_vars = ['access', 'coast_line', 'ownership', 'simple_shape', 'is_marked']
fig, axs = plt.subplots(1, len(binary_vars), figsize=(4*len(binary_vars), 6), sharey=True)

for ax, var in zip(axs, binary_vars):
    subset = df.dropna(subset=['log_price', var])
    sns.violinplot(x=var, y='log_price', data=subset, ax=ax,
                   inner='quartile', cut=0)
    g0 = subset[subset[var] == 0]['log_price']
    g1 = subset[subset[var] == 1]['log_price']
    u, p = stats.mannwhitneyu(g0, g1, alternative='two-sided')
    ax.text(0.5, 0.95, f"U = {u:.1f}\np = {p:.2e}",
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle="round", fc="w"), fontsize=10)

fig.suptitle("Violin Plots for Binary Predictors\n(log(Price per sqm))", y=0.98, fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("output_charts/level_1_violin_binary.png")
plt.close(fig)

# 4) Violin plots for categorical predictors (region & zone)
df['region_group'] = df['region'].map(lambda x: 'AB' if x in ['A','B']
                                     else ('CD' if x in ['C','D'] else 'E'))
def determine_zone(row):
    if row['is_north_forest_steppe'] == 1: return "North"
    if row['is_south_forest_steppe'] == 1: return "South"
    if row['is_steppe'] == 1: return "Steppe"
    return "Other"
df['zone'] = df.apply(determine_zone, axis=1)

fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, (col, color) in zip(axs, [('region_group', 'lightcoral'), ('zone', 'lightseagreen')]):
    subset = df.dropna(subset=['log_price'])
    # Fix: Use color instead of palette
    sns.violinplot(x=col, y='log_price', data=subset, ax=ax,
                   inner='quartile', cut=0, color=color)
    groups = [g['log_price'].values for _, g in subset.groupby(col)]
    F, p = stats.f_oneway(*groups)
    ax.set_title(f"log(Price per sqm) by {col.replace('_',' ').title()}", fontsize=12)
    ax.text(0.05, 0.95, f"F = {F:.2f}\np = {p:.2e}",
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle="round", fc="w"), fontsize=10)

fig.suptitle("Violin Plots for Categorical Predictors\n(log(Price per sqm))", y=0.98, fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("output_charts/level_1_violin_categorical.png")
plt.close(fig)