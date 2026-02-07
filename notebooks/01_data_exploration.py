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
# # 📊 Al-Ranim Real Estate - Data Exploration & Cleaning
# ## Mudon, Dubai | Investor Analysis Foundation
#
# This notebook provides a comprehensive exploration of the Al-Ranim real estate market data,
# covering both **sales transactions** and **rental contracts**. Every finding is framed from
# an **investor's perspective** — what matters for making informed purchase decisions.
#
# **Key Questions Answered:**
# - What does the data look like? What's clean, what's messy?
# - How many actual transactions vs. marketing listings?
# - What location features (park-facing, corner, etc.) can we extract?
# - What does the service charge (3.1 AED/sqft) mean for real yields?

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

from src.config import *
from src.data_loader import load_sales_data, load_rental_data, get_transactions_only, get_contracts_only, get_listings_only
from src.feature_engineering import extract_comment_features, normalize_developer_type, create_sales_features, create_rental_features
from src.visualization import setup_style, format_aed, plot_price_distribution, plot_comparison_bars, plot_correlation_heatmap, plot_location_premium

setup_style()
print("✅ All libraries loaded successfully")

# %% [markdown]
# ---
# ## 1. Data Loading
# We load both CSV files using our data loader which handles latin-1 encoding,
# date parsing, numeric conversion, and **automatically excludes apartment records**.

# %%
sales = load_sales_data()
rental = load_rental_data()

print(f"📦 Sales dataset: {sales.shape[0]:,} rows × {sales.shape[1]} columns")
print(f"📦 Rental dataset: {rental.shape[0]:,} rows × {rental.shape[1]} columns")
print(f"\n🏠 Sales unit types: {sales['unit_type'].value_counts().to_dict()}")
print(f"🏠 Rental unit types: {rental['unit_type'].value_counts().to_dict()}")
print(f"\n✅ Apartments excluded — analysis covers Townhouses & Villas only")

# %%
print("=== SALES DATA - First 5 Rows ===")
sales.head()

# %%
print("=== RENTAL DATA - First 5 Rows ===")
rental.head()

# %% [markdown]
# ---
# ## 2. Data Profiling — Sales
# Understanding the shape, types, and distributions of every column.

# %%
print("=" * 60)
print("SALES DATA PROFILING")
print("=" * 60)
print(f"\nDate range: {sales['custom_date'].min()} to {sales['custom_date'].max()}")
print(f"\nEvidence types (CRITICAL for investors):")
print(sales['evdnc_name'].value_counts().to_string())
print(f"\nUnit types:")
print(sales['unit_type'].value_counts().to_string())
print(f"\nBedrooms:")
print(sales['no_beds'].value_counts().to_string())
print(f"\nFloor levels:")
print(sales['floor_level'].value_counts().to_string())
print(f"\nSub-locations:")
print(sales['sub_loc_2'].value_counts().sort_index().to_string())
print(f"\nSale sequence:")
print(sales['sale_sequence'].value_counts(dropna=False).to_string())

# %%
sales[['total_sales_price_val', 'sales_price_sqft_unit', 'unit_size_sqft', 'plot_size_sqft']].describe().round(0)

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Sales Data — Categorical Distributions', fontsize=16, fontweight='bold')

for ax, col, title in zip(
    axes.flat,
    ['evdnc_name', 'unit_type', 'no_beds', 'floor_level', 'sub_loc_2', 'sale_sequence'],
    ['Evidence Type', 'Unit Type', 'Bedrooms', 'Floor Level', 'Sub-Location', 'Sale Type']
):
    counts = sales[col].value_counts()
    bars = ax.bar(range(len(counts)), counts.values, color=sns.color_palette('husl', len(counts)), edgecolor='white')
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=9)
    ax.set_title(title, fontweight='bold')
    ax.bar_label(bars, padding=2, fontsize=8)

plt.tight_layout()
plt.savefig('../notebooks/fig_01_sales_categoricals.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 3. Data Profiling — Rentals

# %%
print("=" * 60)
print("RENTAL DATA PROFILING")
print("=" * 60)
beds_col = 'beds' if 'beds' in rental.columns else 'no_beds'
print(f"\nDate range: {rental['start_date'].min()} to {rental['start_date'].max()}")
print(f"\nEvidence types:")
print(rental['evdnc_name'].value_counts().to_string())
print(f"\nBedrooms:")
print(rental[beds_col].value_counts().to_string())
print(f"\nSub-locations:")
print(rental['sub_loc_2'].value_counts().sort_index().to_string())
print(f"\nFurnished:")
print(rental['furnished'].value_counts(dropna=False).to_string())

# %%
rent_num_cols = [c for c in ['annualised_rental_price', 'contract_rental_price', 'rent_price_sqft_unit', 'unit_size', 'plot_size'] if c in rental.columns]
rental[rent_num_cols].describe().round(0)

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Rental Data — Categorical Distributions', fontsize=16, fontweight='bold')

rent_cats = ['evdnc_name', 'unit_type', beds_col, 'floor_level', 'sub_loc_2', 'furnished']
rent_titles = ['Evidence Type', 'Unit Type', 'Bedrooms', 'Floor Level', 'Sub-Location', 'Furnished']

for ax, col, title in zip(axes.flat, rent_cats, rent_titles):
    if col in rental.columns:
        counts = rental[col].value_counts()
        bars = ax.bar(range(len(counts)), counts.values, color=sns.color_palette('husl', len(counts)), edgecolor='white')
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=9)
        ax.set_title(title, fontweight='bold')
        ax.bar_label(bars, padding=2, fontsize=8)

plt.tight_layout()
plt.savefig('../notebooks/fig_01_rental_categoricals.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 4. Missing Value Analysis
# Understanding what's missing is crucial — missing data can bias our ML models.

# %%
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Sales missing
sales_missing = sales.isnull().mean().sort_values(ascending=False) * 100
sales_missing_df = sales_missing[sales_missing > 0]
axes[0].barh(range(len(sales_missing_df)), sales_missing_df.values, color='#E63946', edgecolor='white')
axes[0].set_yticks(range(len(sales_missing_df)))
axes[0].set_yticklabels(sales_missing_df.index)
axes[0].set_xlabel('% Missing')
axes[0].set_title('Sales Data — Missing Values', fontweight='bold')
for i, v in enumerate(sales_missing_df.values):
    axes[0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

# Rental missing
rental_missing = rental.isnull().mean().sort_values(ascending=False) * 100
rental_missing_df = rental_missing[rental_missing > 0]
axes[1].barh(range(len(rental_missing_df)), rental_missing_df.values, color='#F18F01', edgecolor='white')
axes[1].set_yticks(range(len(rental_missing_df)))
axes[1].set_yticklabels(rental_missing_df.index)
axes[1].set_xlabel('% Missing')
axes[1].set_title('Rental Data — Missing Values', fontweight='bold')
for i, v in enumerate(rental_missing_df.values):
    axes[1].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('../notebooks/fig_01_missing_values.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Missing Value Decisions
#
# | Column | % Missing | Decision | Rationale |
# |--------|-----------|----------|-----------|
# | `sub_loc_3/4` | ~100% | **Drop** | No useful data |
# | `maid/study/balcony` | ~100% | **Drop** | Empty columns |
# | `custom_view` | ~90% | **Drop** from modeling | Sparse, extracted from comments instead |
# | `developer_type` | ~90% | **Parse** what exists | Extract unit_subtype when available |
# | `plot_size_sqft` | ~38% | **Impute** median by segment | Important feature — impute per beds+floor |
# | `unit_no` | ~95% | **Drop** | Not useful for modeling |
# | Core price/size cols | <1% | **Keep** | Essential features, nearly complete |

# %% [markdown]
# ---
# ## 5. Outlier Detection

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Sales Data — Outlier Detection', fontsize=16, fontweight='bold')

for ax, col, title in zip(
    axes.flat,
    ['total_sales_price_val', 'sales_price_sqft_unit', 'unit_size_sqft', 'plot_size_sqft'],
    ['Total Sale Price (AED)', 'Price per Sqft (AED)', 'Unit Size (sqft)', 'Plot Size (sqft)']
):
    data = sales[col].dropna()
    bp = ax.boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                    medianprops=dict(color='#E63946', linewidth=2))
    ax.set_title(title, fontweight='bold')
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    outliers = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum()
    ax.set_xlabel(f'Outliers: {outliers} ({outliers/len(data)*100:.1f}%)')

plt.tight_layout()
plt.savefig('../notebooks/fig_01_sales_outliers.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Rental Data — Outlier Detection', fontsize=16, fontweight='bold')

for ax, col, title in zip(
    axes.flat,
    ['annualised_rental_price', 'rent_price_sqft_unit', 'unit_size'],
    ['Annual Rent (AED)', 'Rent per Sqft (AED)', 'Unit Size (sqft)']
):
    if col in rental.columns:
        data = rental[col].dropna()
        bp = ax.boxplot(data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='#F18F01', alpha=0.7),
                        medianprops=dict(color='#E63946', linewidth=2))
        ax.set_title(title, fontweight='bold')
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        outliers = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum()
        ax.set_xlabel(f'Outliers: {outliers} ({outliers/len(data)*100:.1f}%)')

plt.tight_layout()
plt.savefig('../notebooks/fig_01_rental_outliers.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Scatter: Price vs Unit Size colored by evidence type
fig, ax = plt.subplots(figsize=(12, 7))
for evd in sales['evdnc_name'].unique():
    mask = sales['evdnc_name'] == evd
    color = PALETTE.get(evd, '#999999')
    ax.scatter(sales.loc[mask, 'unit_size_sqft'], sales.loc[mask, 'total_sales_price_val'],
               alpha=0.4, label=evd, s=30, color=color)
ax.set_xlabel('Unit Size (sqft)')
ax.set_ylabel('Total Sale Price (AED)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
ax.set_title('Sale Price vs Unit Size — by Evidence Type', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('../notebooks/fig_01_price_vs_size.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 6. Feature Engineering
# We extract valuable features from the comments field and normalize developer types.

# %%
# Apply feature engineering
sales = extract_comment_features(sales)
sales = normalize_developer_type(sales)
sales = create_sales_features(sales)

rental = extract_comment_features(rental)
rental = create_rental_features(rental)

print("✅ Feature engineering complete")
print(f"\nSales columns: {sales.shape[1]}")
print(f"Rental columns: {rental.shape[1]}")

# %%
# Location features distribution
location_features = ['is_park_facing', 'is_corner_unit', 'is_single_row', 'is_end_unit',
                     'is_pool_adjacent', 'is_road_facing', 'is_near_amenities']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Sales location features
sales_loc_counts = sales[location_features].sum().sort_values(ascending=True)
axes[0].barh(range(len(sales_loc_counts)), sales_loc_counts.values, color='#2E86AB', edgecolor='white')
axes[0].set_yticks(range(len(sales_loc_counts)))
axes[0].set_yticklabels(sales_loc_counts.index)
axes[0].set_title('Sales — Location Features Extracted from Comments', fontweight='bold')
for i, v in enumerate(sales_loc_counts.values):
    axes[0].text(v + 2, i, f'{int(v)} ({v/len(sales)*100:.1f}%)', va='center', fontsize=9)

# Rental location features
rental_loc_feats = [f for f in location_features if f in rental.columns]
rental_loc_counts = rental[rental_loc_feats].sum().sort_values(ascending=True)
axes[1].barh(range(len(rental_loc_counts)), rental_loc_counts.values, color='#F18F01', edgecolor='white')
axes[1].set_yticks(range(len(rental_loc_counts)))
axes[1].set_yticklabels(rental_loc_counts.index)
axes[1].set_title('Rental — Location Features Extracted from Comments', fontweight='bold')
for i, v in enumerate(rental_loc_counts.values):
    axes[1].text(v + 2, i, f'{int(v)} ({v/len(rental)*100:.1f}%)', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('../notebooks/fig_01_location_features.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Marketing features distribution
marketing_features = ['is_brand_new', 'is_motivated_seller', 'is_hot_deal', 'has_big_plot',
                      'is_upgraded', 'is_rented', 'has_mortgage_mention']
mkt_feats = [f for f in marketing_features if f in sales.columns]
sales_mkt_counts = sales[mkt_feats].sum().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(range(len(sales_mkt_counts)), sales_mkt_counts.values, color='#2BA84A', edgecolor='white')
ax.set_yticks(range(len(sales_mkt_counts)))
ax.set_yticklabels(sales_mkt_counts.index)
ax.set_title('Sales — Marketing Features Extracted from Comments', fontweight='bold')
for i, v in enumerate(sales_mkt_counts.values):
    ax.text(v + 2, i, f'{int(v)} ({v/len(sales)*100:.1f}%)', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('../notebooks/fig_01_marketing_features.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 7. Location Feature Deep Dive
# **This is what investors care about** — how does property position affect value?

# %%
# Cross-tab: location features by sub-location
loc_by_subloc = sales.groupby('sub_loc_2')[location_features].sum()
fig, ax = plt.subplots(figsize=(14, 6))
loc_by_subloc.plot(kind='bar', ax=ax, edgecolor='white')
ax.set_title('Location Features by Sub-Location (Sales)', fontweight='bold')
ax.set_xlabel('Sub-Location')
ax.set_ylabel('Count')
plt.xticks(rotation=45, ha='right')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('../notebooks/fig_01_location_by_subloc.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Price by park-facing status
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sales
sales['park_label'] = sales['is_park_facing'].map({1: 'Park Facing', 0: 'Not Park Facing'})
sns.boxplot(data=sales, x='park_label', y='total_sales_price_val', ax=axes[0],
            palette=['#F4A261', '#2BA84A'])
axes[0].set_title('Sale Price: Park Facing vs Not', fontweight='bold')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
axes[0].set_xlabel('')

# Add medians text
for i, label in enumerate(['Not Park Facing', 'Park Facing']):
    subset = sales[sales['park_label'] == label]['total_sales_price_val']
    axes[0].text(i, subset.median(), f'Median: AED {subset.median():,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Rental
rental['park_label'] = rental['is_park_facing'].map({1: 'Park Facing', 0: 'Not Park Facing'})
sns.boxplot(data=rental, x='park_label', y='annualised_rental_price', ax=axes[1],
            palette=['#F4A261', '#2BA84A'])
axes[1].set_title('Annual Rent: Park Facing vs Not', fontweight='bold')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
axes[1].set_xlabel('')

for i, label in enumerate(['Not Park Facing', 'Park Facing']):
    subset = rental[rental['park_label'] == label]['annualised_rental_price']
    if len(subset) > 0:
        axes[1].text(i, subset.median(), f'Median: AED {subset.median():,.0f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('../notebooks/fig_01_park_facing_impact.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Corner unit and single row price impact
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sales['corner_label'] = sales['is_corner_unit'].map({1: 'Corner Unit', 0: 'Non-Corner'})
sns.boxplot(data=sales, x='corner_label', y='total_sales_price_val', ax=axes[0],
            palette=['#2E86AB', '#E63946'])
axes[0].set_title('Sale Price: Corner Unit vs Non-Corner', fontweight='bold')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
axes[0].set_xlabel('')

sales['row_label'] = sales['is_single_row'].map({1: 'Single Row', 0: 'Non-Single Row'})
sns.boxplot(data=sales, x='row_label', y='total_sales_price_val', ax=axes[1],
            palette=['#2E86AB', '#9B59B6'])
axes[1].set_title('Sale Price: Single Row vs Non-Single Row', fontweight='bold')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
axes[1].set_xlabel('')

plt.tight_layout()
plt.savefig('../notebooks/fig_01_corner_singlerow.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 8. Transaction vs Listing Split (CRITICAL FOR INVESTORS)
#
# **This is the most important distinction in the data.** Active Listings are asking prices
# set by sellers/agents — they are aspirational. Only Oqood (off-plan registrations) and
# Title Deed records represent actual money-changed-hands transactions.
#
# **Investors must use transaction data for decision-making, not listing prices.**

# %%
# Split summary
print("=" * 60)
print("TRANSACTION vs LISTING SPLIT — SALES")
print("=" * 60)
for src in ['transaction', 'listing', 'pending']:
    subset = sales[sales['data_source'] == src]
    print(f"\n{src.upper()}: {len(subset):,} records")
    if len(subset) > 0:
        print(f"  Median price: AED {subset['total_sales_price_val'].median():,.0f}")
        print(f"  Mean price:   AED {subset['total_sales_price_val'].mean():,.0f}")
        print(f"  Median PSF:   AED {subset['sales_price_sqft_unit'].median():,.0f}")

print(f"\n{'=' * 60}")
print("TRANSACTION vs LISTING SPLIT — RENTALS")
print("=" * 60)
for src in ['contract', 'listing']:
    subset = rental[rental['data_source'] == src]
    print(f"\n{src.upper()}: {len(subset):,} records")
    if len(subset) > 0:
        print(f"  Median annual rent: AED {subset['annualised_rental_price'].median():,.0f}")

# %%
# Side-by-side price distributions
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Sales: Transaction vs Listing
transactions = get_transactions_only(sales)
listings = get_listings_only(sales)

axes[0].hist(transactions['total_sales_price_val'].dropna(), bins=30, alpha=0.6,
             color='#2BA84A', label=f'Transactions (n={len(transactions)})', density=True, edgecolor='white')
axes[0].hist(listings['total_sales_price_val'].dropna(), bins=30, alpha=0.4,
             color='#F4A261', label=f'Listings (n={len(listings)})', density=True, edgecolor='white')
axes[0].set_title('Sale Price Distribution: Transactions vs Listings', fontweight='bold')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
axes[0].legend(fontsize=10)

# Calculate listing premium
trans_median = transactions['total_sales_price_val'].median()
list_median = listings['total_sales_price_val'].median()
premium_pct = (list_median - trans_median) / trans_median * 100
axes[0].axvline(trans_median, color='#2BA84A', linestyle='--', linewidth=2, label=f'Trans. Median: AED {trans_median:,.0f}')
axes[0].axvline(list_median, color='#F18F01', linestyle='--', linewidth=2, label=f'List. Median: AED {list_median:,.0f}')
axes[0].legend(fontsize=9)

# Rental: Contract vs Listing
contracts = get_contracts_only(rental)
rental_listings = get_listings_only(rental)

axes[1].hist(contracts['annualised_rental_price'].dropna(), bins=25, alpha=0.6,
             color='#2BA84A', label=f'Contracts (n={len(contracts)})', density=True, edgecolor='white')
axes[1].hist(rental_listings['annualised_rental_price'].dropna(), bins=25, alpha=0.4,
             color='#F4A261', label=f'Listings (n={len(rental_listings)})', density=True, edgecolor='white')
axes[1].set_title('Annual Rent Distribution: Contracts vs Listings', fontweight='bold')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('../notebooks/fig_01_transaction_vs_listing.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n⚠️  LISTING PREMIUM: Listings are {premium_pct:+.1f}% above actual transaction prices")
print(f"    Transaction median: AED {trans_median:,.0f}")
print(f"    Listing median:     AED {list_median:,.0f}")

# %%
# Violin plots by evidence type
fig, ax = plt.subplots(figsize=(14, 6))
order = ['Title Deed', 'Oqood', 'Pending Sales (SPA/MOU)', 'Active Listings']
existing = [o for o in order if o in sales['evdnc_name'].values]
sns.violinplot(data=sales, x='evdnc_name', y='total_sales_price_val', order=existing,
               palette=[PALETTE.get(e, '#999') for e in existing], ax=ax, inner='box')
ax.set_title('Price Distribution by Evidence Type — Violin Plot', fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
ax.set_xlabel('')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('../notebooks/fig_01_violin_evidence.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 9. Service Charge Analysis
#
# **Service charge: 3.1 AED per sqft of built-up area per year.**
# This is a mandatory cost that directly eats into investor returns.

# %%
# Service charge impact table
print("=" * 60)
print("SERVICE CHARGE IMPACT (3.1 AED/sqft/year)")
print("=" * 60)

for beds in sorted(sales['no_beds'].dropna().unique()):
    subset = sales[sales['no_beds'] == beds]
    median_size = subset['unit_size_sqft'].median()
    sc = median_size * SERVICE_CHARGE_PER_SQFT
    print(f"\n{beds}BR Townhouse:")
    print(f"  Median unit size:      {median_size:,.0f} sqft")
    print(f"  Annual service charge: AED {sc:,.0f}")
    print(f"  Monthly equivalent:    AED {sc/12:,.0f}")

# %%
# Service charge distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(sales['annual_service_charge'].dropna(), bins=30, color='#E63946',
             alpha=0.7, edgecolor='white')
axes[0].set_title('Distribution of Annual Service Charge (Sales)', fontweight='bold')
axes[0].set_xlabel('Annual Service Charge (AED)')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(format_aed))

# Gross vs Net yield preview
trans = get_transactions_only(sales)
contr = get_contracts_only(rental)

if len(trans) > 0 and len(contr) > 0:
    # Simple yield preview using medians
    med_price = trans['total_sales_price_val'].median()
    med_rent = contr['annualised_rental_price'].median()
    med_size = trans['unit_size_sqft'].median()
    med_sc = med_size * SERVICE_CHARGE_PER_SQFT

    gross_yield = (med_rent / med_price) * 100
    net_yield = ((med_rent - med_sc) / med_price) * 100

    yields = [gross_yield, net_yield]
    labels = [f'Gross Yield\n{gross_yield:.2f}%', f'Net Yield\n(after service charge)\n{net_yield:.2f}%']
    colors = ['#2BA84A', '#E63946']
    bars = axes[1].bar([0, 1], yields, color=colors, edgecolor='white', width=0.5)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_ylabel('Yield %')
    axes[1].set_title('Yield Preview: Gross vs Net (Community Median)', fontweight='bold')
    axes[1].bar_label(bars, fmt='%.2f%%', padding=5, fontsize=11, fontweight='bold')

    print(f"\n📊 YIELD PREVIEW (using medians of actual transactions & contracts):")
    print(f"   Median purchase price: AED {med_price:,.0f}")
    print(f"   Median annual rent:    AED {med_rent:,.0f}")
    print(f"   Median service charge: AED {med_sc:,.0f}")
    print(f"   Gross yield:           {gross_yield:.2f}%")
    print(f"   Net yield:             {net_yield:.2f}%")
    print(f"   Service charge impact: {gross_yield - net_yield:.2f}% reduction")

plt.tight_layout()
plt.savefig('../notebooks/fig_01_service_charge.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 10. Correlation Analysis

# %%
# Sales correlations
numeric_sales = ['total_sales_price_val', 'sales_price_sqft_unit', 'unit_size_sqft',
                 'plot_size_sqft', 'beds_numeric', 'is_g_plus_2', 'is_park_facing',
                 'is_corner_unit', 'is_single_row', 'is_resale', 'months_since_start']
valid_cols = [c for c in numeric_sales if c in sales.columns and sales[c].notna().sum() > 10]

fig, ax = plt.subplots(figsize=(12, 10))
corr = sales[valid_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Sales Data — Feature Correlations', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('../notebooks/fig_01_sales_correlations.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Rental correlations
numeric_rental = ['annualised_rental_price', 'rent_price_sqft_unit']
size_col = 'unit_size' if 'unit_size' in rental.columns else 'unit_size_sqft'
numeric_rental += [size_col, 'beds_numeric', 'is_park_facing', 'is_corner_unit', 'is_single_row']
if 'is_g_plus_2' in rental.columns:
    numeric_rental.append('is_g_plus_2')
if 'is_furnished' in rental.columns:
    numeric_rental.append('is_furnished')

valid_rcols = [c for c in numeric_rental if c in rental.columns and rental[c].notna().sum() > 10]

fig, ax = plt.subplots(figsize=(10, 8))
corr_r = rental[valid_rcols].corr()
mask_r = np.triu(np.ones_like(corr_r, dtype=bool))
sns.heatmap(corr_r, mask=mask_r, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Rental Data — Feature Correlations', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('../notebooks/fig_01_rental_correlations.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 11. Timeline Analysis

# %%
# Records over time
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Sales by month
if 'year_month' in sales.columns:
    sales_monthly = sales.groupby(['year_month', 'data_source']).size().unstack(fill_value=0)
    sales_monthly.plot(kind='bar', stacked=True, ax=axes[0],
                       color=['#F4A261', '#9B59B6', '#2BA84A'], edgecolor='white')
    axes[0].set_title('Sales Records by Month & Data Source', fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].legend(title='Source')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Rental by month
if 'year_month' in rental.columns:
    rental_monthly = rental.groupby(['year_month', 'data_source']).size().unstack(fill_value=0)
    rental_monthly.plot(kind='bar', stacked=True, ax=axes[1],
                        color=['#F4A261', '#1B3A5C'], edgecolor='white')
    axes[1].set_title('Rental Records by Month & Data Source', fontweight='bold')
    axes[1].set_ylabel('Count')
    axes[1].legend(title='Source')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../notebooks/fig_01_timeline.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 12. Price Distribution by Sub-Location

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Sales by sub-location
subloc_order_sales = [s for s in SUBLOC_ORDER if s in sales['sub_loc_2'].values]
sns.boxplot(data=sales, x='sub_loc_2', y='total_sales_price_val', order=subloc_order_sales,
            palette=[PALETTE.get(s, '#999') for s in subloc_order_sales], ax=axes[0])
axes[0].set_title('Sale Price by Sub-Location', fontweight='bold')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
axes[0].set_xlabel('')
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Rental by sub-location
subloc_order_rental = [s for s in SUBLOC_ORDER if s in rental['sub_loc_2'].values]
sns.boxplot(data=rental, x='sub_loc_2', y='annualised_rental_price', order=subloc_order_rental,
            palette=[PALETTE.get(s, '#999') for s in subloc_order_rental], ax=axes[1])
axes[1].set_title('Annual Rent by Sub-Location', fontweight='bold')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
axes[1].set_xlabel('')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../notebooks/fig_01_subloc_prices.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 13. Data Summary & Key Findings
#
# ### Key Findings for Investors:
#
# 1. **Transaction vs Listing Gap**: Listing prices are inflated above actual transaction prices.
#    Always use transaction data (Oqood/Title Deed) for valuation, not asking prices.
#
# 2. **Limited Transaction Data**: Only ~400 actual sale transactions and ~265 rental contracts.
#    ML models must use cross-validation carefully.
#
# 3. **Location Matters**: Park-facing and corner properties show price premiums visible even
#    in raw data. This will be quantified precisely in the ML notebooks.
#
# 4. **Service Charge Impact**: At 3.1 AED/sqft, service charges reduce gross yield measurably.
#    All yield figures in this analysis are presented net of service charges.
#
# 5. **Homogeneous Market**: Nearly all properties are 3BR or 4BR townhouses in G+1 or G+2
#    configurations. This helps model accuracy but limits diversity of investment options.
#
# 6. **Sub-Area Variation**: Al Ranim 1-8 sub-areas show different price levels, transaction
#    volumes, and data maturity. Areas 5-8 are newer (mostly Oqood-stage).

# %%
# Final summary statistics
print("=" * 60)
print("FINAL DATA SUMMARY")
print("=" * 60)
print(f"\nSales Data:  {sales.shape[0]:,} rows, {sales.shape[1]} columns")
print(f"  Transactions: {len(get_transactions_only(sales)):,}")
print(f"  Listings:     {len(get_listings_only(sales)):,}")
print(f"  Price range:  AED {sales['total_sales_price_val'].min():,.0f} — AED {sales['total_sales_price_val'].max():,.0f}")

print(f"\nRental Data: {rental.shape[0]:,} rows, {rental.shape[1]} columns")
print(f"  Contracts:    {len(get_contracts_only(rental)):,}")
print(f"  Listings:     {len(get_listings_only(rental)):,}")
print(f"  Rent range:   AED {rental['annualised_rental_price'].min():,.0f} — AED {rental['annualised_rental_price'].max():,.0f}")

print(f"\nFeatures engineered: {len(location_features)} location + {len(mkt_feats)} marketing")
print(f"Service charge calculated: 3.1 AED/sqft × unit size")

# %%
# Save cleaned data for downstream notebooks
sales.to_csv('../notebooks/sales_cleaned.csv', index=False)
rental.to_csv('../notebooks/rental_cleaned.csv', index=False)
print("✅ Cleaned datasets saved to notebooks/sales_cleaned.csv and rental_cleaned.csv")
print("   These will be used by all downstream analysis notebooks.")
