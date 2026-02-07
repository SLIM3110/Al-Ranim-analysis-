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
# # 📈 Trend Analysis & 12-Month Projections
# ## Al-Ranim, Mudon, Dubai
#
# This notebook analyses price, rent, and yield trends over time, and projects them
# **12 months forward** using Holt's Exponential Smoothing with bootstrap confidence intervals.

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

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import linregress

from src.config import *
from src.data_loader import load_sales_data, load_rental_data, get_transactions_only, get_contracts_only, get_listings_only
from src.feature_engineering import extract_comment_features, normalize_developer_type, create_sales_features, create_rental_features
from src.visualization import setup_style, format_aed

setup_style()
print("✅ Libraries loaded")

# %%
sales = load_sales_data()
sales = extract_comment_features(sales)
sales = create_sales_features(sales)

rental = load_rental_data()
rental = extract_comment_features(rental)
rental = create_rental_features(rental)

transactions = get_transactions_only(sales)
listings = get_listings_only(sales)
contracts = get_contracts_only(rental)

print(f"Sales: {len(transactions)} transactions, {len(listings)} listings")
print(f"Rental: {len(contracts)} contracts")

# %% [markdown]
# ---
# ## 1. Sale Price Trends

# %%
# Monthly transaction statistics
trans_monthly = transactions.groupby('year_month').agg(
    count=('total_sales_price_val', 'count'),
    median_price=('total_sales_price_val', 'median'),
    mean_price=('total_sales_price_val', 'mean'),
    median_psf=('sales_price_sqft_unit', 'median'),
).reset_index().sort_values('year_month')

list_monthly = listings.groupby('year_month').agg(
    count=('total_sales_price_val', 'count'),
    median_price=('total_sales_price_val', 'median'),
).reset_index().sort_values('year_month')

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Price trend
if len(trans_monthly) > 1:
    axes[0].plot(trans_monthly['year_month'], trans_monthly['median_price'], 'o-',
                 color='#2BA84A', linewidth=2, markersize=8, label='Transaction Median')
if len(list_monthly) > 1:
    axes[0].plot(list_monthly['year_month'], list_monthly['median_price'], 's--',
                 color='#F4A261', linewidth=2, markersize=6, label='Listing Median')
axes[0].set_ylabel('Median Price (AED)')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
axes[0].set_title('Monthly Sale Price Trend: Transactions vs Listings', fontweight='bold')
axes[0].legend()
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Volume
if len(trans_monthly) > 1:
    axes[1].bar(trans_monthly['year_month'], trans_monthly['count'],
                color='#2E86AB', alpha=0.7, label='Transaction Count', edgecolor='white')
axes[1].set_ylabel('Transaction Count')
axes[1].set_title('Monthly Transaction Volume', fontweight='bold')
axes[1].legend()
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../notebooks/fig_05_price_trend.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Price trend by bedrooms
fig, ax = plt.subplots(figsize=(14, 6))
for beds in sorted(transactions['beds_numeric'].dropna().unique()):
    sub = transactions[transactions['beds_numeric'] == beds]
    monthly = sub.groupby('year_month')['total_sales_price_val'].median().reset_index()
    monthly = monthly.sort_values('year_month')
    if len(monthly) > 1:
        ax.plot(monthly['year_month'], monthly['total_sales_price_val'], 'o-',
                linewidth=2, markersize=6, label=f'{int(beds)}BR')
ax.set_ylabel('Median Price (AED)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
ax.set_title('Price Trend: 3BR vs 4BR Transactions', fontweight='bold')
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../notebooks/fig_05_price_by_beds.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 2. Rental Trends

# %%
if 'year_month' in contracts.columns:
    rent_monthly = contracts.groupby('year_month').agg(
        count=('annualised_rental_price', 'count'),
        median_rent=('annualised_rental_price', 'median'),
    ).reset_index().sort_values('year_month')

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    if len(rent_monthly) > 1:
        axes[0].plot(rent_monthly['year_month'], rent_monthly['median_rent'], 'o-',
                     color='#1B3A5C', linewidth=2, markersize=8)
    axes[0].set_ylabel('Median Annual Rent (AED)')
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
    axes[0].set_title('Monthly Rental Contract Trend', fontweight='bold')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    if len(rent_monthly) > 1:
        axes[1].bar(rent_monthly['year_month'], rent_monthly['count'],
                    color='#1B3A5C', alpha=0.7, edgecolor='white')
    axes[1].set_ylabel('Contract Count')
    axes[1].set_title('Monthly Rental Contract Volume', fontweight='bold')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('../notebooks/fig_05_rental_trend.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ---
# ## 3. Sub-Location Appreciation Comparison

# %%
fig, ax = plt.subplots(figsize=(14, 6))
for loc in SUBLOC_ORDER:
    sub = transactions[transactions['sub_loc_2'] == loc]
    if len(sub) > 10:
        monthly = sub.groupby('year_month')['sales_price_sqft_unit'].median().reset_index().sort_values('year_month')
        if len(monthly) > 2:
            ax.plot(monthly['year_month'], monthly['sales_price_sqft_unit'], 'o-',
                    linewidth=2, markersize=5, label=loc, color=PALETTE.get(loc, '#999'))

ax.set_ylabel('Median Price per Sqft (AED)')
ax.set_title('Price/Sqft Trend by Sub-Location', fontweight='bold')
ax.legend(bbox_to_anchor=(1.02, 1), fontsize=9)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../notebooks/fig_05_subloc_trend.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 4. Location Premium Trends (Park-Facing)

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Park-facing price premium over time
for pf, label, color in [(1, 'Park Facing', '#2BA84A'), (0, 'Not Park Facing', '#F4A261')]:
    sub = transactions[transactions['is_park_facing'] == pf]
    monthly = sub.groupby('year_month')['total_sales_price_val'].median().reset_index().sort_values('year_month')
    if len(monthly) > 1:
        axes[0].plot(monthly['year_month'], monthly['total_sales_price_val'], 'o-',
                     linewidth=2, markersize=6, label=label, color=color)
axes[0].set_title('Sale Price: Park Facing vs Not (Over Time)', fontweight='bold')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
axes[0].legend()
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Park-facing rent premium over time
for pf, label, color in [(1, 'Park Facing', '#2BA84A'), (0, 'Not Park Facing', '#F4A261')]:
    sub = contracts[contracts['is_park_facing'] == pf]
    if 'year_month' in sub.columns:
        monthly = sub.groupby('year_month')['annualised_rental_price'].median().reset_index().sort_values('year_month')
        if len(monthly) > 1:
            axes[1].plot(monthly['year_month'], monthly['annualised_rental_price'], 'o-',
                         linewidth=2, markersize=6, label=label, color=color)
axes[1].set_title('Annual Rent: Park Facing vs Not (Over Time)', fontweight='bold')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
axes[1].legend()
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../notebooks/fig_05_park_premium_trend.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 5. 12-Month Price Projection
#
# Using Holt's Exponential Smoothing (linear trend, no seasonality — appropriate for
# limited data). We also apply bootstrap resampling for confidence intervals.

# %%
def project_series(monthly_df, value_col, periods=12, title=""):
    """Project a monthly time series forward with confidence intervals."""
    ts = monthly_df.set_index('year_month')[value_col].sort_index()
    ts.index = pd.RangeIndex(len(ts))

    if len(ts) < 4:
        print(f"⚠️ Insufficient data points ({len(ts)}) for projection: {title}")
        return None

    # Fit Holt's exponential smoothing
    try:
        model = ExponentialSmoothing(ts.values, trend='add', seasonal=None)
        fit = model.fit(optimized=True)
        forecast = fit.forecast(periods)
    except Exception:
        # Fallback to linear regression
        x = np.arange(len(ts))
        slope, intercept, _, _, _ = linregress(x, ts.values)
        x_future = np.arange(len(ts), len(ts) + periods)
        forecast = slope * x_future + intercept

    # Bootstrap confidence intervals
    n_boot = 200
    boot_forecasts = np.zeros((n_boot, periods))
    for b in range(n_boot):
        idx = np.random.choice(len(ts), size=len(ts), replace=True)
        boot_ts = ts.values[np.sort(idx)]
        try:
            bmodel = ExponentialSmoothing(boot_ts, trend='add', seasonal=None)
            bfit = bmodel.fit(optimized=True)
            boot_forecasts[b] = bfit.forecast(periods)
        except Exception:
            x = np.arange(len(boot_ts))
            sl, ic, _, _, _ = linregress(x, boot_ts)
            x_f = np.arange(len(boot_ts), len(boot_ts) + periods)
            boot_forecasts[b] = sl * x_f + ic

    ci_low = np.percentile(boot_forecasts, 10, axis=0)
    ci_high = np.percentile(boot_forecasts, 90, axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    months_hist = monthly_df['year_month'].tolist()
    months_future = [f'M+{i+1}' for i in range(periods)]
    all_months = months_hist + months_future

    ax.plot(range(len(ts)), ts.values, 'o-', color='#2E86AB', linewidth=2, markersize=6, label='Historical')
    ax.plot(range(len(ts), len(ts)+periods), forecast, 's--', color='#E63946', linewidth=2, markersize=6, label='Projection')
    ax.fill_between(range(len(ts), len(ts)+periods), ci_low, ci_high,
                     alpha=0.2, color='#E63946', label='80% Confidence Band')
    ax.axvline(len(ts)-0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xticks(range(len(all_months)))
    ax.set_xticklabels(all_months, rotation=45, ha='right', fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
    ax.set_title(f'12-Month Projection: {title}', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    return fig, forecast, ci_low, ci_high

# %%
# Project sale prices
if len(trans_monthly) > 3:
    fig, fc_price, lo, hi = project_series(trans_monthly, 'median_price', 12, 'Median Transaction Price')
    if fig:
        plt.savefig('../notebooks/fig_05_price_projection.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"📈 12-month price projection: AED {fc_price[-1]:,.0f}")
        print(f"   80% CI: AED {lo[-1]:,.0f} — AED {hi[-1]:,.0f}")

# %%
# Project rents
if 'year_month' in contracts.columns:
    rent_monthly_sorted = rent_monthly.sort_values('year_month')
    if len(rent_monthly_sorted) > 3:
        fig2, fc_rent, lo_r, hi_r = project_series(rent_monthly_sorted, 'median_rent', 12, 'Median Contract Rent')
        if fig2:
            plt.savefig('../notebooks/fig_05_rent_projection.png', dpi=150, bbox_inches='tight')
            plt.show()
            print(f"📈 12-month rent projection: AED {fc_rent[-1]:,.0f}")
            print(f"   80% CI: AED {lo_r[-1]:,.0f} — AED {hi_r[-1]:,.0f}")

# %% [markdown]
# ---
# ## 6. Absorption & Supply Analysis

# %%
# Listing volume and transaction volume over time
fig, ax = plt.subplots(figsize=(14, 6))

if len(trans_monthly) > 1:
    ax.bar(range(len(trans_monthly)), trans_monthly['count'], color='#2BA84A',
           alpha=0.7, label='Transactions', edgecolor='white')
if len(list_monthly) > 1:
    ax2 = ax.twinx()
    ax2.plot(range(len(list_monthly)), list_monthly['count'], 'o-', color='#E63946',
             linewidth=2, markersize=6, label='Listings')
    ax2.set_ylabel('Listing Count', color='#E63946')

ax.set_xticks(range(max(len(trans_monthly), len(list_monthly))))
labels = trans_monthly['year_month'].tolist() if len(trans_monthly) >= len(list_monthly) else list_monthly['year_month'].tolist()
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('Transaction Count', color='#2BA84A')
ax.set_title('Market Absorption: Transactions vs Listings Over Time', fontweight='bold')
ax.legend(loc='upper left')
if len(list_monthly) > 1:
    ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('../notebooks/fig_05_absorption.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Linear trend summary
if len(trans_monthly) > 2:
    x = np.arange(len(trans_monthly))
    slope, intercept, r_val, p_val, std_err = linregress(x, trans_monthly['median_price'])
    annual_appreciation = (slope * 12 / trans_monthly['median_price'].mean()) * 100

    print("=" * 60)
    print("TREND SUMMARY")
    print("=" * 60)
    print(f"Sale price trend: {slope:+,.0f} AED/month (R²={r_val**2:.3f})")
    print(f"Annualised appreciation: {annual_appreciation:+.1f}%")

if 'rent_monthly' in dir() and len(rent_monthly) > 2:
    x = np.arange(len(rent_monthly))
    slope_r, _, r_val_r, _, _ = linregress(x, rent_monthly['median_rent'])
    annual_rent_change = (slope_r * 12 / rent_monthly['median_rent'].mean()) * 100
    print(f"Rent trend: {slope_r:+,.0f} AED/month (R²={r_val_r**2:.3f})")
    print(f"Annualised rent change: {annual_rent_change:+.1f}%")

# %% [markdown]
# ---
# ## Key Takeaways for Investors
#
# 1. **Price trajectory**: The 12-month projection shows the expected price direction
#    with confidence bands. Use the conservative (lower) bound for purchase planning.
#
# 2. **Rent outlook**: Rental trends indicate whether yield compression or expansion
#    is likely over the next year.
#
# 3. **Location premiums**: Park-facing premiums have been [stable/growing/shrinking] —
#    this directly affects whether paying extra for park views is a good long-term bet.
#
# 4. **Market absorption**: The ratio of transactions to listings signals market health.
#    Falling ratios suggest buyer's market; rising ratios suggest seller's market.
