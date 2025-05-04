import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# 0) Setup
os.makedirs('output_charts', exist_ok=True)
df = pd.read_csv('data/data_level_5.csv')

# 1) Transformations
df['log_price']          = np.log(df['price_per_sqm'])
df['log_area']           = np.log(df['area'])
df['log_distance_capital'] = np.log(df['distance_capital'])

# 2) Price distribution (2×2)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Top-left: histogram natural
axs[0, 0].hist(df['price_per_sqm'], bins=10, color='skyblue', edgecolor='black')
axs[0, 0].set(title="Histogram of Price per sqm\n(Natural Scale)",
              xlabel="Price per sqm", ylabel="Count")

# Top-right: histogram log + normal
axs[0, 1].hist(df['log_price'], bins=10, density=True, color='lightgreen', edgecolor='black')
mu, std = stats.norm.fit(df['log_price'])
x = np.linspace(df['log_price'].min(), df['log_price'].max(), 100)
axs[0, 1].plot(x, stats.norm.pdf(x, mu, std), 'r--', lw=2)
axs[0, 1].set(title="Histogram of log(Price per sqm)\nwith Normal Fit",
              xlabel="log(Price per sqm)", ylabel="Density")

# Bottom-left: KDE natural + normal
sns.kdeplot(df['price_per_sqm'], ax=axs[1, 0], fill=True, color='blue', label='KDE')
mu_nat, std_nat = stats.norm.fit(df['price_per_sqm'])
x_nat = np.linspace(df['price_per_sqm'].min(), df['price_per_sqm'].max(), 100)
axs[1, 0].plot(x_nat, stats.norm.pdf(x_nat, mu_nat, std_nat), 'r--', lw=2, label='Normal Fit')
axs[1, 0].set(title="KDE of Price per sqm\n(Natural Scale)",
              xlabel="Price per sqm", ylabel="Density")
axs[1, 0].legend()

# Bottom-right: KDE log + normal
sns.kdeplot(df['log_price'], ax=axs[1, 1], fill=True, color='purple', label='KDE')
axs[1, 1].plot(x, stats.norm.pdf(x, mu, std), 'r--', lw=2, label='Normal Fit')
axs[1, 1].set(title="KDE of log(Price per sqm)",
              xlabel="log(Price per sqm)", ylabel="Density")
axs[1, 1].legend()

fig.tight_layout()
fig.savefig("output_charts/level_5_price_distribution.png")
plt.close(fig)

# 3) Scatter plots (1×2, log–log)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for ax, var, label in zip(
    axs,
    ['log_area', 'log_distance_capital'],
    ['log(Area)', 'log(Distance from Capital)']
):
    ax.scatter(df[var], df['log_price'], color='teal', alpha=0.7)
    r, p = stats.pearsonr(df[var], df['log_price'])
    ax.set(title=f"log(Price) vs. {label}", xlabel=label, ylabel="log(Price per sqm)")
    ax.text(0.05, 0.95, f"r = {r:.2f}\np = {p:.2e}",
            transform=ax.transAxes, va='top', bbox=dict(boxstyle="round", fc="w"))
fig.tight_layout()
fig.savefig("output_charts/level_5_scatterplots.png")
plt.close(fig)

# 4) Violin for ownership
fig, ax = plt.subplots(figsize=(6, 6))
sns.violinplot(x='ownership', y='log_price', data=df,
               inner='quartile', cut=0, color='skyblue', density_norm='width', ax=ax)
ax.set(title="log(Price per sqm) by Ownership", xlabel="Ownership (1=Private)", ylabel="log(Price)")
# Mann–Whitney U
g0 = df[df['ownership'] == 0]['log_price']
g1 = df[df['ownership'] == 1]['log_price']
u, p = stats.mannwhitneyu(g0, g1, alternative='two-sided')
ax.text(0.5, 0.95, f"U = {u:.1f}\np = {p:.2e}",
        transform=ax.transAxes, ha='center', va='top',
        bbox=dict(boxstyle="round", fc="w"), fontsize=12)
fig.tight_layout()
fig.savefig("output_charts/level_5_violin_ownership.png")
plt.close(fig)
