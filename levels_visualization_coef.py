import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ensure output directory exists
os.makedirs('level_visualizations', exist_ok=True)

# Function to extract coefficients from a summary file
def extract_coefficients(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract the coefficient table section using a more flexible pattern
    coef_pattern = r'-{3,}\n(.*?)\n={3,}'
    coef_match = re.search(coef_pattern, content, re.DOTALL)

    if not coef_match:
        # Try alternative patterns if the first one fails
        coef_pattern = r'coef\s+std err(.*?)Prob'
        coef_match = re.search(coef_pattern, content, re.DOTALL)

    if not coef_match:
        print(f"Warning: Could not extract coefficients from {file_path}")
        return {}

    # Parse the coefficient rows
    coef_rows = coef_match.group(1).strip().split('\n')
    coefficients = {}

    for row in coef_rows:
        parts = row.split()
        if len(parts) >= 2:
            try:
                var_name = parts[0]
                coef_value = float(parts[1])
                coefficients[var_name] = coef_value
            except ValueError:
                # Skip rows that don't have proper coefficient values
                continue

    print(f"Extracted {len(coefficients)} coefficients from {file_path}")
    return coefficients

# Get all level summary files
level_files = []
for file in os.listdir('primary_models'):
    if file.startswith('level') and file.endswith('_summary.txt'):
        match = re.search(r'level(\d+)_summary', file)
        if match:
            level_num = int(match.group(1))
            level_files.append((level_num, os.path.join('primary_models', file)))
        else:
            print(f"Warning: Could not parse level number from {file}")

# Sort by level number
level_files.sort()
print(f"Found {len(level_files)} level files: {[f[0] for f in level_files]}")

# Extract coefficients from each level
all_coefficients = {}
for level_num, file_path in level_files:
    coeffs = extract_coefficients(file_path)
    if coeffs:
        all_coefficients[level_num] = coeffs

# Get all unique coefficient names across all levels
all_coef_names = set()
for level_coeffs in all_coefficients.values():
    all_coef_names.update(level_coeffs.keys())

# Remove 'Intercept' as it's typically much larger and can skew the visualization
if 'Intercept' in all_coef_names:
    all_coef_names.remove('Intercept')

# Print coefficient names for debugging
print(f"Found {len(all_coef_names)} unique coefficients")
if len(all_coef_names) == 0:
    print("No coefficients found. Check your summary file format.")
    exit(1)

# Convert set to list for DataFrame index
all_coef_names = sorted(list(all_coef_names))

# Create a DataFrame with coefficients across levels
levels = sorted(all_coefficients.keys())
coef_df = pd.DataFrame(index=all_coef_names, columns=levels)

for level in levels:
    for coef in all_coef_names:
        coef_df.loc[coef, level] = all_coefficients[level].get(coef, 0)

# Print the dataframe for debugging
print("Coefficient DataFrame:")
print(coef_df.head())

# Plot coefficients across levels
plt.figure(figsize=(14, 10))

# Generate a distinct color for each coefficient using a colormap that works well for many categories
color_map = plt.cm.get_cmap('tab20' if len(all_coef_names) <= 20 else 'gist_rainbow')
colors = [color_map(i/len(all_coef_names)) for i in range(len(all_coef_names))]

# Create markers for different coefficients to improve visibility
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
if len(markers) < len(all_coef_names):
    # Repeat markers if we have more coefficients than marker types
    markers = markers * (len(all_coef_names) // len(markers) + 1)

for i, coef in enumerate(coef_df.index):
    values = coef_df.loc[coef].values
    x_vals = levels

    # Skip if all values are zero
    if np.all(values == 0):
        print(f"Skipping {coef} - all values are zero")
        continue

    marker = markers[i % len(markers)]

    # Plot the actual coefficient values
    line, = plt.plot(x_vals, values, marker=marker, label=coef,
                    color=colors[i], linewidth=2, markersize=8)

    # Add dashed lines where coefficients become zero
    for j in range(1, len(values)):
        if (values[j-1] != 0 and values[j] == 0) or (values[j-1] == 0 and values[j] != 0):
            plt.plot([x_vals[j-1], x_vals[j]], [values[j-1], values[j]],
                    '--', color=line.get_color(), alpha=0.7, linewidth=1.5)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

# Add grid, labels, and legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Model Level', fontsize=14)
plt.ylabel('Coefficient Value', fontsize=14)
plt.title('Coefficient Values Across Different Model Levels', fontsize=16, pad=20)
plt.xticks(levels, fontsize=12)

# Create a more manageable legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, frameon=True,
          fancybox=True, shadow=True, ncol=1 if len(all_coef_names) <= 15 else 2)

plt.tight_layout()
plt.savefig('level_visualizations/coefficient_evolution.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualization saved to 'level_visualizations/coefficient_evolution.png'")