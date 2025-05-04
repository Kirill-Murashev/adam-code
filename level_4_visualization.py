import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Ensure output directory exists
os.makedirs('output_charts', exist_ok=True)

# 1) Read the Level III data
df = pd.read_csv('data/data_level_4.csv')
df['log_price'] = np.log(df['price_per_sqm'])

# 2) Price distribution (2×2)
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Top-left: Histogram (natural scale)
axs[0, 0].hist(df['price_per_sqm'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axs[0, 0].set(title="Histogram of Price per sqm\n(Natural Scale)",
              xlabel="Price per sqm", ylabel="Frequency")

# Top-right: Histogram (log scale) + normal fit
n, bins, _ = axs[0, 1].hist(df['log_price'], bins=50, density=True,
                            color='lightgreen', edgecolor='black', alpha=0.7)
mu, std = stats.norm.fit(df['log_price'])
x = np.linspace(df['log_price'].min(), df['log_price'].max(), 100)
axs[0, 1].plot(x, stats.norm.pdf(x, mu, std), 'r--', lw=2)
axs[0, 1].set(title="Histogram of log(Price) with Normal Fit",
              xlabel="log(Price per sqm)", ylabel="Density")

# Bottom-left: KDE (natural) + normal fit
sns.kdeplot(df['price_per_sqm'], ax=axs[1, 0], fill=True, color='blue', label='KDE')
mu_nat, std_nat = stats.norm.fit(df['price_per_sqm'])
x_nat = np.linspace(df['price_per_sqm'].min(), df['price_per_sqm'].max(), 100)
axs[1, 0].plot(x_nat, stats.norm.pdf(x_nat, mu_nat, std_nat), 'r--', label='Normal Fit')
axs[1, 0].set(title="KDE of Price per sqm\n(Natural Scale)",
              xlabel="Price per sqm", ylabel="Density")
axs[1, 0].legend()

# Bottom-right: KDE (log) + normal fit
sns.kdeplot(df['log_price'], ax=axs[1, 1], fill=True, color='purple', label='KDE')
axs[1, 1].plot(x, stats.norm.pdf(x, mu, std), 'r--', label='Normal Fit')
axs[1, 1].set(title="KDE of log(Price per sqm)",
              xlabel="log(Price per sqm)", ylabel="Density")
axs[1, 1].legend()

fig.tight_layout()
fig.savefig("output_charts/level_4_price_distribution.png")
plt.close(fig)

# 3) Scatter plots (2×2, log–log)
df['log_area']              = np.log(df['area'])
df['log_distance_capital'] = np.log(df['distance_capital'])
df['log_distance_elevator'] = np.log(df['distance_elevator'])
df['log_crop_yield']       = np.log(df['crop_yield'])

scatter_vars = {
    'log_area': 'log(Area)',
    'log_distance_capital': 'log(Distance from Capital)',
    'log_distance_elevator': 'log(Distance from Elevator)',
    'log_crop_yield': 'log(Crop Yield)'
}

fig, axs = plt.subplots(2, 2, figsize=(14, 12))
for ax, (var, label) in zip(axs.flatten(), scatter_vars.items()):
    ax.scatter(df[var], df['log_price'], alpha=0.5, color='teal')
    r, p = stats.pearsonr(df[var], df['log_price'])
    ax.set(title=f"log(Price) vs. {label}",
           xlabel=label, ylabel="log(Price per sqm)")
    ax.text(0.05, 0.95, f"r = {r:.2f}\np = {p:.2e}",
            transform=ax.transAxes, va='top', bbox=dict(boxstyle="round", fc="w"))
fig.tight_layout()
fig.savefig("output_charts/level_4_scatterplots.png")
plt.close(fig)

# 4) Violin plots for binary predictors (no hue, color only)
binary_vars = [
    'ownership', 'simple_shape'
]
fig, axs = plt.subplots(1, len(binary_vars), figsize=(4*len(binary_vars), 6), sharey=True)

for ax, var in zip(axs, binary_vars):
    sns.violinplot(x=var, y='log_price', data=df, ax=ax,
                   inner='quartile', hue=var, cut=0,
                   palette='pastel', density_norm='width',
                   legend=False)
    ax.set_title(var, fontsize=12)
    g0 = df[df[var] == 0]['log_price']
    g1 = df[df[var] == 1]['log_price']
    u, p = stats.mannwhitneyu(g0, g1, alternative='two-sided')
    ax.text(0.5, 0.95, f"U = {u:.1f}\np = {p:.2e}",
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle="round", fc="w"), fontsize=10)

fig.suptitle("Violin Plots for Binary Predictors\n(log(Price per sqm))", y=0.98, fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("output_charts/level_4_violin_binary.png")
plt.close(fig)
