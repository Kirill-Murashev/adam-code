import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ensure output directory exists
os.makedirs('level_visualizations', exist_ok=True)

# Get all SHAP importance files
shap_files = []
for file in os.listdir('primary_models'):
    if file.startswith('shap_importance_l') and file.endswith('.csv'):
        match = re.search(r'shap_importance_l(\d+)\.csv', file)
        if match:
            level_num = int(match.group(1))
            shap_files.append((level_num, os.path.join('primary_models', file)))
        else:
            print(f"Warning: Could not parse level number from {file}")

# Sort by level number
shap_files.sort()
print(f"Found {len(shap_files)} SHAP importance files: {[f[0] for f in shap_files]}")

# Columns to visualize
columns = ['mean_abs_shap', 'pct_importance', 'explained_variance_share']
line_styles = {
    'mean_abs_shap': '-',
    'pct_importance': '--',
    'explained_variance_share': ':'
}

# Read all files and organize data by feature and column
all_features = set()
feature_data = {}

for level_num, file_path in shap_files:
    try:
        df = pd.read_csv(file_path)
        features = df['feature'].tolist()
        all_features.update(features)

        for feature in features:
            if feature not in feature_data:
                feature_data[feature] = {col: {} for col in columns}

            for col in columns:
                value = df.loc[df['feature'] == feature, col].values[0]
                feature_data[feature][col][level_num] = value

        print(f"Processed {file_path}: {len(features)} features")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Convert feature set to sorted list
all_features = sorted(list(all_features))
levels = sorted([level for level, _ in shap_files])

# Create a figure for each column
for col_idx, column in enumerate(columns):
    plt.figure(figsize=(14, 10))

    # Generate a distinct color for each feature
    color_map = plt.cm.get_cmap('tab20' if len(all_features) <= 20 else 'gist_rainbow')
    colors = [color_map(i/len(all_features)) for i in range(len(all_features))]

    # Create markers for different features
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    if len(markers) < len(all_features):
        markers = markers * (len(all_features) // len(markers) + 1)

    for i, feature in enumerate(all_features):
        # Collect values across levels
        values = []
        x_vals = []

        for level in levels:
            if level in feature_data[feature][column]:
                x_vals.append(level)
                values.append(feature_data[feature][column][level])

        if not values:
            print(f"Skipping {feature} for {column} - no data")
            continue

        marker = markers[i % len(markers)]
        line_style = line_styles[column]

        # Plot the values
        line, = plt.plot(x_vals, values, marker=marker, linestyle=line_style,
                        label=feature, color=colors[i], linewidth=2, markersize=8)

    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Add grid, labels, and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Model Level', fontsize=14)
    plt.ylabel(column.replace('_', ' ').title(), fontsize=14)
    plt.title(f'{column.replace("_", " ").title()} Across Different Model Levels', fontsize=16, pad=20)
    plt.xticks(levels, fontsize=12)

    # Create a more manageable legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, frameon=True,
              fancybox=True, shadow=True, ncol=1 if len(all_features) <= 15 else 2)

    plt.tight_layout()
    plt.savefig(f'level_visualizations/{column}_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to 'level_visualizations/{column}_evolution.png'")

# Now create a combined visualization with all metrics for each feature
# This will show how different metrics compare for each feature

for feature in all_features:
    plt.figure(figsize=(12, 8))

    for col_idx, column in enumerate(columns):
        # Collect values across levels
        values = []
        x_vals = []

        for level in levels:
            if level in feature_data[feature][column]:
                x_vals.append(level)
                values.append(feature_data[feature][column][level])

        if not values:
            continue

        line_style = line_styles[column]

        # Plot the values with different line styles for each metric
        plt.plot(x_vals, values, marker='o', linestyle=line_style,
                label=column.replace('_', ' ').title(), linewidth=2, markersize=8)

    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Add grid, labels, and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Model Level', fontsize=14)
    plt.ylabel('Importance Value', fontsize=14)
    plt.title(f'Importance Metrics for "{feature}" Across Model Levels', fontsize=16, pad=20)
    plt.xticks(levels, fontsize=12)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(f'level_visualizations/metrics_for_{feature}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Feature metrics visualization saved to 'level_visualizations/metrics_for_{feature}.png'")