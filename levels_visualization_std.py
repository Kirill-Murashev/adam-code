import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ensure output directory exists
os.makedirs('level_visualizations', exist_ok=True)

# Function to extract standard errors from a summary file
def extract_std_errors(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract the coefficient table section
    coef_pattern = r'-{3,}\n(.*?)\n={3,}'
    coef_match = re.search(coef_pattern, content, re.DOTALL)

    if not coef_match:
        # Try alternative patterns if the first one fails
        coef_pattern = r'coef\s+std err(.*?)Prob'
        coef_match = re.search(coef_pattern, content, re.DOTALL)

    if not coef_match:
        print(f"Warning: Could not extract standard errors from {file_path}")
        return {}

    # Parse the coefficient rows
    coef_rows = coef_match.group(1).strip().split('\n')
    std_errors = {}

    for row in coef_rows:
        parts = row.split()
        if len(parts) >= 3:  # Need at least var_name, coef, and std_err
            try:
                var_name = parts[0]
                std_err = float(parts[2])  # Standard error is in the third column
                std_errors[var_name] = std_err
            except ValueError:
                # Skip rows that don't have proper values
                continue

    print(f"Extracted {len(std_errors)} standard errors from {file_path}")
    return std_errors

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

# Extract standard errors from each level
all_std_errors = {}
for level_num, file_path in level_files:
    std_errs = extract_std_errors(file_path)
    if std_errs:
        all_std_errors[level_num] = std_errs

# Get all unique variable names across all levels
all_var_names = set()
for level_std_errs in all_std_errors.values():
    all_var_names.update(level_std_errs.keys())

# Remove 'Intercept' if desired (optional)
if 'Intercept' in all_var_names:
    all_var_names.remove('Intercept')

# Print variable names for debugging
print(f"Found {len(all_var_names)} unique variables")
if len(all_var_names) == 0:
    print("No standard errors found. Check your summary file format.")
    exit(1)

# Convert set to list for DataFrame index
all_var_names = sorted(list(all_var_names))

# Create a DataFrame with standard errors across levels
levels = sorted(all_std_errors.keys())
stderr_df = pd.DataFrame(index=all_var_names, columns=levels)

for level in levels:
    for var in all_var_names:
        stderr_df.loc[var, level] = all_std_errors[level].get(var, 0)

# Print the dataframe for debugging
print("Standard Error DataFrame:")
print(stderr_df.head())

# Plot standard errors across levels
plt.figure(figsize=(14, 10))

# Generate a distinct color for each variable
color_map = plt.cm.get_cmap('tab20' if len(all_var_names) <= 20 else 'gist_rainbow')
colors = [color_map(i/len(all_var_names)) for i in range(len(all_var_names))]

# Create markers for different variables to improve visibility
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
if len(markers) < len(all_var_names):
    # Repeat markers if we have more variables than marker types
    markers = markers * (len(all_var_names) // len(markers) + 1)

for i, var in enumerate(stderr_df.index):
    values = stderr_df.loc[var].values
    x_vals = levels

    # Skip if all values are zero
    if np.all(values == 0):
        print(f"Skipping {var} - all values are zero")
        continue

    marker = markers[i % len(markers)]

    # Plot the actual standard error values
    line, = plt.plot(x_vals, values, marker=marker, label=var,
                    color=colors[i], linewidth=2, markersize=8)

    # Add dashed lines where variables appear/disappear from the model
    for j in range(1, len(values)):
        if (values[j-1] != 0 and values[j] == 0) or (values[j-1] == 0 and values[j] != 0):
            plt.plot([x_vals[j-1], x_vals[j]], [values[j-1], values[j]],
                    '--', color=line.get_color(), alpha=0.7, linewidth=1.5)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

# Add grid, labels, and legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Model Level', fontsize=14)
plt.ylabel('Standard Error', fontsize=14)
plt.title('Standard Errors Across Different Model Levels', fontsize=16, pad=20)
plt.xticks(levels, fontsize=12)

# Create a more manageable legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, frameon=True,
          fancybox=True, shadow=True, ncol=1 if len(all_var_names) <= 15 else 2)

plt.tight_layout()
plt.savefig('level_visualizations/std_error_evolution.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualization saved to 'level_visualizations/std_error_evolution.png'")