import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("TkAgg")  # Use TkAgg backend
import matplotlib.pyplot as plt
import numpy as np

# Set seaborn style to poster
# sns.set_style("whitegrid")
sns.set_context("talk")

# Create a DataFrame from the table data
data = {
    'Configuration': ['36 States - Original', '22 States - Original', '16 States - Original',
                      '36 States - Prosthesis', '22 States - Prosthesis', '16 States - Prosthesis'],
    'IL': [10000, 10000, 3110.1, 10000, 10000, 6604],
    'RL': [2230, 1108, 450, 4671.2, 331, 1207.7]
}

# For values that are 10000+, we'll use 10000 for visualization purposes
df = pd.DataFrame(data)

# Melt the DataFrame to convert it to a format suitable for seaborn
melted_df = pd.melt(df, id_vars=['Configuration'],
                    value_vars=['IL', 'RL'],
                    var_name='Agent', value_name='Steps')

# Create the bar chart
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Configuration', y='Steps', hue='Agent', data=melted_df)

# Add horizontal dotted line at 10000
plt.axhline(y=10000, color='r', linestyle='--', label='Baseline')

# Customize the plot
plt.title('IL vs RL Agents - Number of Steps Before Falling', fontsize=18, fontweight='bold')
plt.xlabel('Configuration', fontsize=12)
plt.ylabel('Number of Steps', fontsize=16)
plt.xticks(rotation=15, fontsize=16)
# plt.legend(title='Agent Type', framealpha=1.0)

# Annotate the bars with their values
for i, p in enumerate(ax.patches):
    height = p.get_height()
    if height >= 10000:
        ax.text(p.get_x() + p.get_width()/2., 9500, '10000+',
                ha='center', va='bottom', fontsize=12)
    elif height > 0:
        ax.text(p.get_x() + p.get_width()/2., height + 100, str(height),
                ha='center', va='bottom', fontsize=12)
plt.xlabel('')
ax.get_legend().remove()
sns.despine()
# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('compare_step.png', dpi=300, bbox_inches='tight')
plt.show()

# If you need to save the figure to a file:
# plt.savefig('il_vs_rl_comparison.png', dpi=300, bbox_inches='tight')