import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure the 'images' subfolder exists
if not os.path.exists('images'):
    os.makedirs('images')

# Number of observations
n = 500

# Define candidate model complexities (number of parameters)
ks = np.arange(1, 11)  # Models with 1 to 10 parameters

# Synthetic log-likelihood values: assume improvement with diminishing returns
LL = 150 + 50 * (1 - np.exp(-0.3 * ks))  # Synthetic log-likelihood values

# Calculate AIC and BIC for each candidate model
AIC = -2 * LL + 2 * ks
BIC = -2 * LL + ks * np.log(n)

# Calculate the difference between BIC and AIC
diff = BIC - AIC

# Plot AIC and BIC as functions of model complexity
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(ks, AIC, marker='o', label='AIC', color='skyblue')
ax1.plot(ks, BIC, marker='s', label='BIC', color='salmon')
ax1.set_xlabel('Number of Parameters (Model Complexity)')
ax1.set_ylabel('Information Criterion Value')
ax1.set_title('AIC and BIC vs Model Complexity')
ax1.legend()
ax1.grid(True)
fig1.tight_layout()
fig1.savefig('images/AIC_BIC_vs_model_complexity.png')
plt.close(fig1)

# Plot the difference between BIC and AIC
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(ks, diff, marker='d', color='purple')
ax2.set_xlabel('Number of Parameters (Model Complexity)')
ax2.set_ylabel('BIC - AIC')
ax2.set_title('Difference between BIC and AIC vs Model Complexity')
ax2.axhline(0, color='black', linestyle='--')
ax2.grid(True)
fig2.tight_layout()
fig2.savefig('images/BIC_minus_AIC_vs_model_complexity.png')
plt.close(fig2)
