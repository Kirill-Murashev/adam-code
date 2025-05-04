import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Ensure the 'output_charts' subfolder exists
if not os.path.exists('output_charts'):
    os.makedirs('output_charts')

# 1) Read the data from CSV
df = pd.read_csv('data/data_level_2.csv')

# Create a log-transformed column for price_per_sqm
df['log_price'] = np.log(df['price_per_sqm'])

##############################################
# 2) Combined 2x2 Chart for Price Distribution
##############################################
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Top Left: Histogram (natural scale)
axs[0, 0].hist(df['price_per_sqm'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axs[0, 0].set_title("Histogram of Price per sqm\n(Natural Scale)")
axs[0, 0].set_xlabel("Price per sqm")
axs[0, 0].set_ylabel("Frequency")

# Top Right: Histogram (log scale) with overlaid normal curve
n, bins, patches = axs[0, 1].hist(df['log_price'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7,
                                  density=True)
mu, std = stats.norm.fit(df['log_price'])
xmin, xmax = axs[0, 1].get_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
axs[0, 1].plot(x, p, 'r--', linewidth=2)
axs[0, 1].set_title("Histogram of log(Price per sqm)\nwith Normal Curve")
axs[0, 1].set_xlabel("log(Price per sqm)")
axs[0, 1].set_ylabel("Density")

# Bottom Left: KDE plot (natural scale) with fitted normal curve
sns.kdeplot(df['price_per_sqm'], ax=axs[1, 0], color='blue', fill=True, label='KDE')
mu_nat, std_nat = stats.norm.fit(df['price_per_sqm'])
x_nat = np.linspace(df['price_per_sqm'].min(), df['price_per_sqm'].max(), 100)
p_nat = stats.norm.pdf(x_nat, mu_nat, std_nat)
axs[1, 0].plot(x_nat, p_nat, 'r--', linewidth=2, label='Normal Fit')
axs[1, 0].set_title("KDE of Price per sqm\n(Natural Scale)")
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
fig.savefig("output_charts/level_2_price_distribution.png")
plt.close(fig)

###################################################
# 3) Combined 2x2 Chart for Scatter Plots (log-log)
###################################################
# Log-transform continuous predictors
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
    ax.scatter(df[var], df['log_price'], alpha=0.5, color='teal')
    ax.set_xlabel(label)
    ax.set_ylabel("log(Price per sqm)")
    ax.set_title(f"log(Price) vs. {label}")

    r, p_value = stats.pearsonr(df[var], df['log_price'])
    ax.text(0.05, 0.95, f"r = {r:.2f}\np = {p_value:.3e}", transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))

fig.tight_layout()
fig.savefig("output_charts/level_2_scatterplots.png")
plt.close(fig)

###############################################################
# 4) Combined Violin Plots for Binary Predictors (log_price)
###############################################################
# Binary predictors for Level II
binary_vars = [
    'access', 'ownership', 'simple_shape', 'is_marked',
    'is_north_forest_steppe', 'is_south_forest_steppe', 'is_steppe'
]

# Create violin plots with updated seaborn parameters
fig, axs = plt.subplots(1, len(binary_vars), figsize=(4 * len(binary_vars), 6), sharey=True)

for ax, var in zip(axs, binary_vars):
    sns.violinplot(
        x=var,
        y='log_price',
        data=df,
        ax=ax,
        inner='quartile',
        hue=var,  # assign hue to avoid palette warning
        palette='pastel',
        density_norm='width',  # replace scale='width'
        cut=0,
        legend=False  # suppress legend
    )
    ax.set_title(var, fontsize=12)

    # Mann–Whitney U test
    group0 = df[df[var] == 0]['log_price']
    group1 = df[df[var] == 1]['log_price']
    u_stat, p_val = stats.mannwhitneyu(group0, group1, alternative='two-sided')
    ax.text(
        0.5, 0.95,
        f"U = {u_stat:.1f}\np = {p_val:.2e}",
        transform=ax.transAxes,
        ha='center', va='top',
        bbox=dict(boxstyle="round", fc="w"),
        fontsize=10
    )

fig.suptitle("Violin Plots for Binary Predictors (log(Price per sqm))", y=0.98, fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("output_charts/level_2_violin_binary_fixed.png")
plt.close(fig)
