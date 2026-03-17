"""Generate Scenario 4 corruption robustness charts from existing CSV data."""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

scenario_dir = os.path.join(config.RESULTS_DIR, "scenario_4_corruption")
csv_path = os.path.join(scenario_dir, "summary_table.csv")
df = pd.read_csv(csv_path)

# Parse accuracy to float
df['Acc'] = df['Accuracy'].str.replace('%', '').astype(float)
# Parse robustness
df['Rob'] = pd.to_numeric(df['Relative Robustness'], errors='coerce')

models = df['Model'].unique()
corruptions = df[df['Corruption'] != 'Clean']['Corruption'].unique()

# ── Plot 1: Grouped bar chart — Accuracy under each corruption ──
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(corruptions))
width = 0.22
colors = ['#2196F3', '#4CAF50', '#FF9800']

for i, model in enumerate(models):
    model_df = df[(df['Model'] == model) & (df['Corruption'] != 'Clean')]
    accs = [model_df[model_df['Corruption'] == c]['Acc'].values[0] for c in corruptions]
    bars = ax.bar(x + i * width, accs, width, label=model, color=colors[i], edgecolor='white')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{acc:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Corruption Type', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Corruption Robustness — Accuracy Under Different Corruptions', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(corruptions, rotation=15, ha='right')
ax.legend(fontsize=10)
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_path = os.path.join(scenario_dir, "corruption_accuracy_comparison.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {save_path}")

# ── Plot 2: Relative Robustness heatmap ──
fig, ax = plt.subplots(figsize=(10, 4))
rob_data = []
for model in models:
    row = []
    for corr in corruptions:
        val = df[(df['Model'] == model) & (df['Corruption'] == corr)]['Rob'].values[0]
        row.append(val)
    rob_data.append(row)

rob_array = np.array(rob_data)
im = ax.imshow(rob_array, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

ax.set_xticks(np.arange(len(corruptions)))
ax.set_yticks(np.arange(len(models)))
ax.set_xticklabels(corruptions, rotation=20, ha='right', fontsize=10)
ax.set_yticklabels(models, fontsize=10)

for i in range(len(models)):
    for j in range(len(corruptions)):
        text_col = 'white' if rob_array[i, j] < 0.35 else 'black'
        ax.text(j, i, f'{rob_array[i,j]:.3f}', ha='center', va='center',
                fontsize=10, fontweight='bold', color=text_col)

ax.set_title('Relative Robustness (Corrupted / Clean Accuracy)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8, label='Relative Robustness')
plt.tight_layout()
save_path = os.path.join(scenario_dir, "relative_robustness_heatmap.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {save_path}")

# ── Plot 3: Per-model accuracy drop chart ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for idx, model in enumerate(models):
    ax = axes[idx]
    model_df = df[df['Model'] == model]
    clean_acc = model_df[model_df['Corruption'] == 'Clean']['Acc'].values[0]

    corr_labels = []
    drops = []
    accs = []
    for corr in corruptions:
        acc = model_df[model_df['Corruption'] == corr]['Acc'].values[0]
        corr_labels.append(corr)
        drops.append(clean_acc - acc)
        accs.append(acc)

    bars = ax.barh(corr_labels, accs, color=colors[idx], alpha=0.8, edgecolor='white')
    ax.axvline(x=clean_acc, color='red', linestyle='--', linewidth=1.5, label=f'Clean ({clean_acc:.1f}%)')

    for bar, acc, drop in zip(bars, accs, drops):
        ax.text(acc + 0.5, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}% (↓{drop:.1f})', va='center', fontsize=9)

    ax.set_xlabel('Accuracy (%)')
    ax.set_title(model, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 110)
    ax.grid(axis='x', alpha=0.3)

fig.suptitle('Accuracy Drop Per Corruption Type', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_path = os.path.join(scenario_dir, "per_model_accuracy_drop.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {save_path}")

# ── Plot 4: Gaussian noise severity curve ──
fig, ax = plt.subplots(figsize=(8, 5))
sigmas = [0, 0.05, 0.1, 0.2]
for i, model in enumerate(models):
    model_df = df[df['Model'] == model]
    clean_acc = model_df[model_df['Corruption'] == 'Clean']['Acc'].values[0]
    gauss_accs = [clean_acc]
    for s in [0.05, 0.1, 0.2]:
        acc = model_df[model_df['Corruption'] == f'Gaussian σ={s}']['Acc'].values[0]
        gauss_accs.append(acc)
    ax.plot(sigmas, gauss_accs, 'o-', label=model, color=colors[i], linewidth=2, markersize=8)

ax.set_xlabel('Gaussian Noise σ', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Accuracy vs. Gaussian Noise Severity', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(sigmas)
plt.tight_layout()
save_path = os.path.join(scenario_dir, "gaussian_noise_severity.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {save_path}")

print("\n✅ All S4 charts generated!")
