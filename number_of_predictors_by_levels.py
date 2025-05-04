import matplotlib.pyplot as plt

# Data for the chart
levels = [1, 2, 3, 4, 5]
variables = [14, 11, 9, 6, 3]

# Create the line chart
plt.figure(figsize=(10, 6))
plt.plot(levels, variables, marker='o', linewidth=2, markersize=10,
         color='#1f77b4', markerfacecolor='white', markeredgewidth=2)

# Add annotations for each point
for x, y in zip(levels, variables):
    plt.annotate(f'{y}',
                xy=(x, y),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=12)

# Customize the chart
plt.title('Number of Meaningful Variables by Aggregation Level', fontsize=14)
plt.xlabel('Aggregation Level', fontsize=12)
plt.ylabel('Number of Variables', fontsize=12)
plt.xticks(levels)
plt.yticks(range(0, max(variables)+2, 2))
plt.grid(True, linestyle='--', alpha=0.7)

# Set y-axis to start from 0
plt.ylim(bottom=0)

# Adjust layout and save
plt.tight_layout()
plt.savefig("output_charts/variables_by_level.png")
plt.close()