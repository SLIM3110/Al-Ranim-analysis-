# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python (Al-Ranim)
#     language: python
#     name: al-ranim-venv
# ---

# %% [markdown]
# # 🎯 Market Segmentation — Investment Tiers & Anomalies
# ## Al-Ranim, Mudon, Dubai
#
# Three unsupervised ML models discover hidden structure in the market:
#
# | Model | How It Thinks | Investor Insight |
# |-------|--------------|-----------------|
# | **K-Means** | Properties cluster into distinct tiers | Investment tier classification |
# | **DBSCAN** | Density-based grouping + outlier detection | Find anomalous deals |
# | **PCA** | Reduce complexity to core dimensions | What really drives differentiation? |

# %%
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from src.config import *
from src.data_loader import load_sales_data, get_transactions_only, get_listings_only
from src.feature_engineering import extract_comment_features, normalize_developer_type, create_sales_features
from src.visualization import setup_style, format_aed

setup_style()
print("✅ Libraries loaded")

# %% [markdown]
# ## 1. Data Preparation

# %%
sales = load_sales_data()
sales = extract_comment_features(sales)
sales = normalize_developer_type(sales)
sales = create_sales_features(sales)

# Features for clustering
cluster_features = ['total_sales_price_val', 'sales_price_sqft_unit', 'unit_size_sqft',
                    'beds_numeric', 'is_g_plus_2', 'is_park_facing', 'is_corner_unit', 'is_single_row']

# Only keep rows with valid price and size
df = sales.dropna(subset=['total_sales_price_val', 'unit_size_sqft', 'sales_price_sqft_unit']).copy()
df['beds_numeric'] = df['beds_numeric'].fillna(3)
df['is_g_plus_2'] = df['is_g_plus_2'].fillna(0)

X_raw = df[cluster_features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"📊 Clustering on {X_scaled.shape[0]} properties × {X_scaled.shape[1]} features")

# %% [markdown]
# ---
# ## 2. K-Means Clustering — Investment Tiers

# %%
# Elbow & silhouette analysis
k_range = range(2, 9)
inertias = []
silhouettes = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(k_range), inertias, 'o-', color='#2E86AB', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method', fontweight='bold')

axes[1].plot(list(k_range), silhouettes, 'o-', color='#2BA84A', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score', fontweight='bold')

best_k = list(k_range)[np.argmax(silhouettes)]
axes[1].axvline(best_k, color='#E63946', linestyle='--', label=f'Best k={best_k}')
axes[1].legend()

plt.tight_layout()
plt.savefig('../notebooks/fig_04_elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Optimal k = {best_k} (silhouette = {max(silhouettes):.3f})")

# %%
# Fit final K-Means
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = km_final.fit_predict(X_scaled)

# Profile each cluster
print("=" * 70)
print(f"K-MEANS CLUSTER PROFILES (k={best_k})")
print("=" * 70)
for c in sorted(df['cluster'].unique()):
    sub = df[df['cluster'] == c]
    print(f"\n🏷️ Cluster {c} — {len(sub):,} properties ({len(sub)/len(df)*100:.1f}%)")
    print(f"   Median price:   AED {sub['total_sales_price_val'].median():,.0f}")
    print(f"   Median PSF:     AED {sub['sales_price_sqft_unit'].median():,.0f}")
    print(f"   Median size:    {sub['unit_size_sqft'].median():,.0f} sqft")
    print(f"   Beds:           {list(sub['beds_numeric'].mode())}")
    print(f"   % Park-facing:  {sub['is_park_facing'].mean()*100:.1f}%")
    print(f"   % Corner:       {sub['is_corner_unit'].mean()*100:.1f}%")
    print(f"   % G+2:          {sub['is_g_plus_2'].mean()*100:.1f}%")
    print(f"   Top sub-locs:   {sub['sub_loc_2'].value_counts().head(3).to_dict()}")

# %%
# Scatter: Price vs Size colored by cluster
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cluster_colors = sns.color_palette('husl', best_k)

for c in range(best_k):
    mask = df['cluster'] == c
    axes[0].scatter(df.loc[mask, 'unit_size_sqft'], df.loc[mask, 'total_sales_price_val'],
                    alpha=0.4, s=30, label=f'Cluster {c}', color=cluster_colors[c])
axes[0].set_xlabel('Unit Size (sqft)')
axes[0].set_ylabel('Total Price (AED)')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
axes[0].set_title('K-Means: Price vs Size by Cluster', fontweight='bold')
axes[0].legend()

# Cluster distribution by sub-location
cluster_subloc = df.groupby(['sub_loc_2', 'cluster']).size().unstack(fill_value=0)
cluster_subloc.plot(kind='bar', stacked=True, ax=axes[1], color=cluster_colors, edgecolor='white')
axes[1].set_title('Cluster Distribution by Sub-Location', fontweight='bold')
axes[1].set_ylabel('Count')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../notebooks/fig_04_kmeans_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Radar chart for cluster profiles
cluster_profiles = df.groupby('cluster')[cluster_features].median()
# Normalize to 0-1 for radar
profile_norm = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min() + 1e-10)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2*np.pi, len(cluster_features), endpoint=False).tolist()
angles += angles[:1]

for c in range(best_k):
    values = profile_norm.loc[c].values.tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {c}', color=cluster_colors[c])
    ax.fill(angles, values, alpha=0.1, color=cluster_colors[c])

ax.set_xticks(angles[:-1])
ax.set_xticklabels([f.replace('_', '\n') for f in cluster_features], fontsize=8)
ax.set_title('Cluster Profiles — Radar Chart', fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('../notebooks/fig_04_radar.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 3. DBSCAN — Anomaly Detection
# *Finds outlier properties that don't fit any natural cluster — potential opportunities or traps.*

# %%
# k-distance graph to choose eps
nn = NearestNeighbors(n_neighbors=2*len(cluster_features))
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_dist = np.sort(distances[:, -1])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(len(k_dist)), k_dist, color='#2E86AB')
ax.set_xlabel('Points (sorted)')
ax.set_ylabel(f'Distance to {2*len(cluster_features)}-th neighbor')
ax.set_title('k-Distance Graph (for DBSCAN eps selection)', fontweight='bold')
# Auto-detect elbow
gradient = np.gradient(k_dist)
elbow_idx = np.argmax(gradient > np.percentile(gradient, 95))
eps_val = k_dist[elbow_idx]
ax.axhline(eps_val, color='#E63946', linestyle='--', label=f'eps ≈ {eps_val:.2f}')
ax.legend()
plt.tight_layout()
plt.savefig('../notebooks/fig_04_kdistance.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Fit DBSCAN
db = DBSCAN(eps=max(eps_val, 0.8), min_samples=2*len(cluster_features))
df['dbscan_label'] = db.fit_predict(X_scaled)

n_clusters_db = len(set(df['dbscan_label'])) - (1 if -1 in df['dbscan_label'].values else 0)
n_outliers = (df['dbscan_label'] == -1).sum()

print(f"DBSCAN: {n_clusters_db} clusters, {n_outliers} outliers ({n_outliers/len(df)*100:.1f}%)")

# Analyze outliers
if n_outliers > 0:
    outliers = df[df['dbscan_label'] == -1]
    normal = df[df['dbscan_label'] != -1]
    print(f"\n📍 OUTLIER PROPERTIES ({n_outliers} total):")
    print(f"   Median price: AED {outliers['total_sales_price_val'].median():,.0f} vs Normal AED {normal['total_sales_price_val'].median():,.0f}")
    print(f"   Median size:  {outliers['unit_size_sqft'].median():,.0f} vs Normal {normal['unit_size_sqft'].median():,.0f} sqft")
    print(f"   % Park-facing: {outliers['is_park_facing'].mean()*100:.1f}% vs Normal {normal['is_park_facing'].mean()*100:.1f}%")

# %%
# Scatter with outliers highlighted
fig, ax = plt.subplots(figsize=(12, 7))
normal_mask = df['dbscan_label'] != -1
ax.scatter(df.loc[normal_mask, 'unit_size_sqft'], df.loc[normal_mask, 'total_sales_price_val'],
           alpha=0.3, s=20, color='#2E86AB', label='Normal')
if n_outliers > 0:
    ax.scatter(df.loc[~normal_mask, 'unit_size_sqft'], df.loc[~normal_mask, 'total_sales_price_val'],
               alpha=0.8, s=60, color='#E63946', marker='x', linewidths=2, label=f'Outliers ({n_outliers})')
ax.set_xlabel('Unit Size (sqft)')
ax.set_ylabel('Total Price (AED)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
ax.set_title('DBSCAN — Outlier Detection', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('../notebooks/fig_04_dbscan_outliers.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 4. PCA — Market Dimensions
# *What are the underlying factors that really differentiate properties?*

# %%
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Scree plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

axes[0].bar(range(1, len(explained)+1), explained*100, color='#2E86AB', edgecolor='white')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained %')
axes[0].set_title('PCA — Scree Plot', fontweight='bold')

axes[1].plot(range(1, len(cumulative)+1), cumulative*100, 'o-', color='#2BA84A', linewidth=2, markersize=8)
axes[1].axhline(80, color='#E63946', linestyle='--', label='80% threshold')
n_comp_80 = np.argmax(cumulative >= 0.80) + 1
axes[1].axvline(n_comp_80, color='#F4A261', linestyle=':', label=f'{n_comp_80} components for 80%')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance %')
axes[1].set_title('PCA — Cumulative Variance', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('../notebooks/fig_04_pca_scree.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ {n_comp_80} components explain ≥80% of variance")
for i in range(min(3, len(explained))):
    print(f"   PC{i+1}: {explained[i]*100:.1f}% variance")

# %%
# PCA Biplot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot points colored by sub_loc_2
for loc in df['sub_loc_2'].unique():
    mask = df['sub_loc_2'] == loc
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.3, s=20,
               label=loc, color=PALETTE.get(loc, '#999'))

# Plot loading arrows
loadings = pca.components_[:2].T
for i, feat in enumerate(cluster_features):
    ax.arrow(0, 0, loadings[i, 0]*5, loadings[i, 1]*5, head_width=0.15,
             head_length=0.1, fc='black', ec='black', alpha=0.7)
    ax.text(loadings[i, 0]*5.5, loadings[i, 1]*5.5, feat.replace('_', '\n'),
            fontsize=7, ha='center', fontweight='bold')

ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}% variance)')
ax.set_title('PCA Biplot — Market Dimensions', fontweight='bold')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('../notebooks/fig_04_pca_biplot.png', dpi=150, bbox_inches='tight')
plt.show()

# PCA loadings interpretation
print("\n📊 PCA LOADINGS INTERPRETATION:")
for pc in range(min(3, len(pca.components_))):
    print(f"\nPC{pc+1} ({explained[pc]*100:.1f}% variance):")
    loadings_pc = pd.Series(pca.components_[pc], index=cluster_features)
    top = loadings_pc.abs().nlargest(3)
    for feat in top.index:
        val = loadings_pc[feat]
        direction = "higher" if val > 0 else "lower"
        print(f"   {feat:30s} → {val:+.3f} ({direction} = higher PC{pc+1})")

# %%
# PCA colored by K-Means cluster
fig, ax = plt.subplots(figsize=(10, 7))
for c in range(best_k):
    mask = df['cluster'] == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.4, s=30,
               label=f'Cluster {c}', color=cluster_colors[c])
ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)')
ax.set_title('PCA — Colored by K-Means Cluster', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('../notebooks/fig_04_pca_clusters.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## Key Takeaways for Investors
#
# 1. **Investment tiers exist**: K-Means reveals distinct property tiers with different
#    price points, sizes, and location characteristics. Each tier suits a different investor profile.
#
# 2. **Anomalies are real**: DBSCAN identifies properties that don't fit the normal pattern.
#    These deserve closer inspection — they may be under- or over-valued.
#
# 3. **Two main dimensions**: PCA shows the market primarily varies along a size-price axis
#    and a location-premium axis. Park-facing and corner features create a distinct dimension
#    separate from size.
