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
# # Al-Ranim Investment Report
# ## Mudon Community, Dubai | February 2026
#
# ---
#
# **Prepared for:** Real Estate Investors
#
# **Data Source:** 2,524 sales records & 2,070 rental records from Eva Real Estate LLC
#
# **Analysis Period:** February 2025 — February 2026 (12 months)
#
# **Methodology:** 15 machine learning models + statistical analysis on actual transaction data
#
# ---
#
# ### How to Read This Report
#
# This report is structured to give you **everything you need to make an investment decision**
# in Al-Ranim, from the big picture down to specific property recommendations.
#
# We separate **actual transactions** (money that changed hands, registered with Dubai Land Department)
# from **listing prices** (what sellers are asking for). This distinction matters because
# **listing prices overstate the real market.**
#
# All yield calculations are **net** — after the 3.1 AED/sqft annual service charge.

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
from scipy.stats import linregress

from src.config import *
from src.data_loader import load_sales_data, load_rental_data, get_transactions_only, get_contracts_only, get_listings_only
from src.feature_engineering import (extract_comment_features, normalize_developer_type,
                                     create_sales_features, create_rental_features, compute_yield_metrics)
from src.visualization import setup_style, format_aed

setup_style()

sales = load_sales_data()
sales = extract_comment_features(sales)
sales = normalize_developer_type(sales)
sales = create_sales_features(sales)

rental = load_rental_data()
rental = extract_comment_features(rental)
rental = create_rental_features(rental)

trans = get_transactions_only(sales)
contracts = get_contracts_only(rental)
listings = get_listings_only(sales)

bcol = 'beds' if 'beds' in contracts.columns else 'no_beds'
if 'beds_numeric' not in contracts.columns:
    contracts = contracts.copy()
    contracts['beds_numeric'] = pd.to_numeric(contracts[bcol], errors='coerce')
scol = 'unit_size' if 'unit_size' in contracts.columns else 'unit_size_sqft'
if 'rent_per_sqft' not in contracts.columns:
    contracts['rent_per_sqft'] = contracts['annualised_rental_price'] / contracts[scol]

# %% [markdown]
# ---
# # Section 1: The Market at a Glance
#
# Al-Ranim is a **townhouse community** within Mudon, developed by **Dubai Properties**.
# It consists of 8 sub-areas (Al-Ranim 1 through 8), with primarily **3-bedroom and
# 4-bedroom townhouses** in G+1 and G+2 configurations.

# %%
fig = plt.figure(figsize=(16, 8))

# Create a clean dashboard layout
ax_main = fig.add_axes([0.05, 0.55, 0.9, 0.4])
ax_main.axis('off')

# Key stats
stats = [
    ('TRANSACTIONS', f'{len(trans)}', 'actual sales recorded'),
    ('MEDIAN PRICE', f'AED {trans["total_sales_price_val"].median()/1e6:.2f}M', 'across all bedrooms'),
    ('RENTAL CONTRACTS', f'{len(contracts)}', 'actual leases signed'),
    ('MEDIAN RENT', f'AED {contracts["annualised_rental_price"].median()/1000:.0f}K/yr', 'annual rent'),
    ('NET YIELD', f'{((contracts["annualised_rental_price"].median() - trans["unit_size_sqft"].median()*3.1) / trans["total_sales_price_val"].median() * 100):.1f}%', 'after service charge'),
    ('MEDIAN PSF', f'AED {trans["sales_price_sqft_unit"].median():,.0f}', 'per square foot'),
]

for i, (title, value, subtitle) in enumerate(stats):
    x = 0.02 + (i % 6) * 0.165
    y = 0.6
    ax_main.text(x, y + 0.25, title, fontsize=9, color='#666666', fontweight='bold')
    ax_main.text(x, y, value, fontsize=16, color='#1B3A5C', fontweight='bold')
    ax_main.text(x, y - 0.2, subtitle, fontsize=8, color='#999999')

ax_main.set_title('Al-Ranim Market Overview — February 2026', fontsize=18, fontweight='bold', pad=20, color='#1B3A5C')

# Bottom charts
ax1 = fig.add_axes([0.07, 0.08, 0.25, 0.38])
ax2 = fig.add_axes([0.4, 0.08, 0.25, 0.38])
ax3 = fig.add_axes([0.73, 0.08, 0.22, 0.38])

# Price distribution
ax1.hist(trans['total_sales_price_val']/1e6, bins=25, color='#2E86AB', edgecolor='white', alpha=0.8)
ax1.set_xlabel('Price (AED Millions)')
ax1.set_title('Transaction Price Distribution', fontweight='bold', fontsize=10)

# Beds split
beds_counts = trans['beds_numeric'].value_counts().sort_index()
bars = ax2.bar([f'{int(b)}BR' for b in beds_counts.index], beds_counts.values,
               color=['#2E86AB', '#E63946'], edgecolor='white')
ax2.set_title('Transactions by Bedrooms', fontweight='bold', fontsize=10)
ax2.bar_label(bars, padding=3)

# Sub-area counts
subloc_counts = trans['sub_loc_2'].value_counts().reindex(
    [s for s in SUBLOC_ORDER if s in trans['sub_loc_2'].values]
)
colors = [PALETTE.get(s, '#999') for s in subloc_counts.index]
ax3.barh(range(len(subloc_counts)), subloc_counts.values, color=colors, edgecolor='white')
ax3.set_yticks(range(len(subloc_counts)))
ax3.set_yticklabels([s.replace('Al Ranim ', 'AR') for s in subloc_counts.index], fontsize=8)
ax3.set_title('By Sub-Area', fontweight='bold', fontsize=10)
ax3.invert_yaxis()

plt.savefig('../notebooks/fig_09_market_overview.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# # Section 2: What Things Actually Cost
#
# **The most common mistake investors make is looking at listing prices.**
# Listings are what sellers *hope* to get. Transactions are what buyers *actually paid*.
#
# Here is what properties in Al-Ranim actually cost based on registered transactions:

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Detailed pricing table
print("=" * 75)
print("WHAT THINGS ACTUALLY COST IN AL-RANIM")
print("=" * 75)
print(f"\n{'':5s}{'Median Price':>15s}{'Median PSF':>12s}{'Size':>10s}{'SC/Year':>12s}{'Transactions':>14s}")
print("-" * 75)

for beds in sorted(trans['beds_numeric'].dropna().unique()):
    t = trans[trans['beds_numeric'] == beds]
    price = t['total_sales_price_val'].median()
    psf = t['sales_price_sqft_unit'].median()
    size = t['unit_size_sqft'].median()
    sc = size * SERVICE_CHARGE_PER_SQFT
    print(f"{int(beds)}BR  AED {price:>12,.0f}  AED {psf:>7,.0f}  {size:>7,.0f}  AED {sc:>8,.0f}  {len(t):>10}")

    for fl in sorted(t['floor_level'].dropna().unique()):
        sub = t[t['floor_level'] == fl]
        if len(sub) >= 5:
            p = sub['total_sales_price_val'].median()
            ps = sub['sales_price_sqft_unit'].median()
            print(f"  {fl}  AED {p:>12,.0f}  AED {ps:>7,.0f}  {sub['unit_size_sqft'].median():>7,.0f}  {'':12s}  {len(sub):>10}")

# Charts
for ax, beds, color in zip(axes, [3.0, 4.0], ['#2E86AB', '#E63946']):
    t = trans[trans['beds_numeric'] == beds]
    ax.hist(t['total_sales_price_val']/1e6, bins=20, color=color, edgecolor='white', alpha=0.8)
    ax.axvline(t['total_sales_price_val'].median()/1e6, color='black', linewidth=2, linestyle='--',
               label=f'Median: AED {t["total_sales_price_val"].median()/1e6:.2f}M')
    ax.set_xlabel('Price (AED Millions)')
    ax.set_ylabel('Count')
    ax.set_title(f'{int(beds)}BR Transaction Prices', fontweight='bold')
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('../notebooks/fig_09_pricing.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# # Section 3: Price Per Square Foot — The True Comparison
#
# When comparing properties of different sizes, **price per square foot (PSF)** is
# the only fair metric. A 3BR at AED 3.2M and a 4BR at AED 4.0M might look very
# different, but on a PSF basis they could be almost identical value.

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PSF by sub-area
subloc_psf = trans.groupby('sub_loc_2')['sales_price_sqft_unit'].median()
subloc_psf = subloc_psf.reindex([s for s in SUBLOC_ORDER if s in subloc_psf.index])
colors = [PALETTE.get(s, '#999') for s in subloc_psf.index]
bars = axes[0].bar(range(len(subloc_psf)), subloc_psf.values, color=colors, edgecolor='white')
axes[0].set_xticks(range(len(subloc_psf)))
axes[0].set_xticklabels([s.replace('Al Ranim ', 'AR') for s in subloc_psf.index], rotation=45, ha='right')
axes[0].set_ylabel('AED / sqft')
axes[0].set_title('PSF by Sub-Area', fontweight='bold')
axes[0].bar_label(bars, [f'{v:,.0f}' for v in subloc_psf.values], padding=3, fontsize=9)

# PSF by beds + floor
cross_psf = trans.groupby(['beds_numeric', 'floor_level'])['sales_price_sqft_unit'].median().unstack()
cross_psf.plot(kind='bar', ax=axes[1], color=['#2BA84A', '#9B59B6'], edgecolor='white')
axes[1].set_title('PSF by Beds + Floor', fontweight='bold')
axes[1].set_ylabel('AED / sqft')
axes[1].set_xticklabels([f'{int(b)}BR' for b in cross_psf.index], rotation=0)

# PSF trend
trans_monthly = trans.groupby('year_month')['sales_price_sqft_unit'].median().reset_index().sort_values('year_month')
if len(trans_monthly) > 2:
    axes[2].plot(trans_monthly['year_month'], trans_monthly['sales_price_sqft_unit'], 'o-',
                  color='#2E86AB', linewidth=2, markersize=6)
    x = np.arange(len(trans_monthly))
    slope, intercept, r_val, _, _ = linregress(x, trans_monthly['sales_price_sqft_unit'])
    axes[2].plot(trans_monthly['year_month'], [slope*i+intercept for i in x],
                  '--', color='#E63946', linewidth=1.5)
    annual_pct = (slope * 12 / trans_monthly['sales_price_sqft_unit'].mean()) * 100
    axes[2].set_title(f'PSF Trend ({annual_pct:+.1f}%/year)', fontweight='bold')
axes[2].set_ylabel('AED / sqft')
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../notebooks/fig_09_psf_overview.png', dpi=150, bbox_inches='tight')
plt.show()

cheapest = subloc_psf.idxmin()
priciest = subloc_psf.idxmax()
spread = subloc_psf.max() - subloc_psf.min()
print(f"\n📊 PSF SUMMARY:")
print(f"   Overall median: AED {trans['sales_price_sqft_unit'].median():,.0f}/sqft")
print(f"   Cheapest area:  {cheapest} at AED {subloc_psf.min():,.0f}/sqft")
print(f"   Priciest area:  {priciest} at AED {subloc_psf.max():,.0f}/sqft")
print(f"   Area spread:    AED {spread:,.0f}/sqft ({spread/subloc_psf.min()*100:.1f}% difference)")

# %% [markdown]
# ---
# # Section 4: What Do Rentals Look Like?
#
# Rental income is the backbone of investment returns. Here is what tenants
# are actually paying based on registered rental contracts (not listing ads).

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

print("=" * 75)
print("RENTAL MARKET — ACTUAL CONTRACT DATA")
print("=" * 75)
print(f"\n{'':5s}{'Median Rent/Yr':>15s}{'Rent PSF':>12s}{'Contracts':>12s}")
print("-" * 50)

for beds in sorted(contracts['beds_numeric'].dropna().unique()):
    c = contracts[contracts['beds_numeric'] == beds]
    rent = c['annualised_rental_price'].median()
    rpsf = c['rent_per_sqft'].median()
    print(f"{int(beds)}BR  AED {rent:>12,.0f}  AED {rpsf:>7,.1f}  {len(c):>8}")

# Rent distribution by beds
for beds, color in [(3.0, '#2E86AB'), (4.0, '#E63946')]:
    c = contracts[contracts['beds_numeric'] == beds]
    axes[0].hist(c['annualised_rental_price']/1000, bins=20, alpha=0.5, color=color,
                 label=f'{int(beds)}BR (med: AED {c["annualised_rental_price"].median()/1000:.0f}K)', edgecolor='white')
axes[0].set_xlabel('Annual Rent (AED Thousands)')
axes[0].set_title('Rent Distribution by Bedrooms', fontweight='bold')
axes[0].legend()

# Rent by sub-area
subloc_rent = contracts.groupby('sub_loc_2')['annualised_rental_price'].median()
subloc_rent = subloc_rent.reindex([s for s in SUBLOC_ORDER if s in subloc_rent.index])
if len(subloc_rent) > 0:
    colors = [PALETTE.get(s, '#999') for s in subloc_rent.index]
    bars = axes[1].bar(range(len(subloc_rent)), subloc_rent.values/1000, color=colors, edgecolor='white')
    axes[1].set_xticks(range(len(subloc_rent)))
    axes[1].set_xticklabels([s.replace('Al Ranim ', 'AR') for s in subloc_rent.index], rotation=45, ha='right')
    axes[1].set_ylabel('AED Thousands / Year')
    axes[1].set_title('Rent by Sub-Area', fontweight='bold')
    axes[1].bar_label(bars, [f'{v:,.0f}K' for v in subloc_rent.values/1000], padding=3, fontsize=9)

# Rent PSF by sub-area
subloc_rpsf = contracts.groupby('sub_loc_2')['rent_per_sqft'].median()
subloc_rpsf = subloc_rpsf.reindex([s for s in SUBLOC_ORDER if s in subloc_rpsf.index])
if len(subloc_rpsf) > 0:
    colors = [PALETTE.get(s, '#999') for s in subloc_rpsf.index]
    bars2 = axes[2].bar(range(len(subloc_rpsf)), subloc_rpsf.values, color=colors, edgecolor='white')
    axes[2].set_xticks(range(len(subloc_rpsf)))
    axes[2].set_xticklabels([s.replace('Al Ranim ', 'AR') for s in subloc_rpsf.index], rotation=45, ha='right')
    axes[2].set_ylabel('AED / sqft / year')
    axes[2].set_title('Rent per Sqft by Sub-Area', fontweight='bold')
    axes[2].bar_label(bars2, [f'{v:,.1f}' for v in subloc_rpsf.values], padding=3, fontsize=9)

plt.tight_layout()
plt.savefig('../notebooks/fig_09_rentals.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# # Section 5: Investment Yields — The Bottom Line
#
# **This is what investors care about most.** How much money does a property actually
# make after costs?
#
# We calculate yields using:
# - **Actual transaction prices** (not listing prices)
# - **Actual rental contracts** (not listing rents)
# - **Net of service charge** (3.1 AED/sqft/year)

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

print("=" * 80)
print("INVESTMENT YIELDS — NET AFTER SERVICE CHARGE")
print("=" * 80)
print(f"\nService charge: 3.1 AED per sqft of built-up area per year")
print(f"\n{'Segment':20s}{'Price':>14s}{'Rent/Yr':>12s}{'SC/Yr':>10s}{'Gross':>8s}{'Net':>8s}{'Count':>8s}")
print("-" * 80)

yield_rows = []
for beds in [3.0, 4.0]:
    t = trans[trans['beds_numeric'] == beds]
    c = contracts[contracts['beds_numeric'] == beds]
    if len(t) > 0 and len(c) > 0:
        price = t['total_sales_price_val'].median()
        rent = c['annualised_rental_price'].median()
        size = t['unit_size_sqft'].median()
        sc = size * SERVICE_CHARGE_PER_SQFT
        gross = rent/price*100
        net = (rent-sc)/price*100
        label = f'{int(beds)}BR Overall'
        print(f"{label:20s} AED {price:>10,.0f} AED {rent:>8,.0f} AED {sc:>6,.0f} {gross:>6.2f}% {net:>6.2f}% {min(len(t),len(c)):>5}")
        yield_rows.append({'Segment': label, 'Gross': gross, 'Net': net})

    # By sub-area
    for loc in SUBLOC_ORDER:
        ts = t[t['sub_loc_2'] == loc]
        cs = c[c['sub_loc_2'] == loc]
        if len(ts) >= 5 and len(cs) >= 5:
            price = ts['total_sales_price_val'].median()
            rent = cs['annualised_rental_price'].median()
            size = ts['unit_size_sqft'].median()
            sc = size * SERVICE_CHARGE_PER_SQFT
            gross = rent/price*100
            net = (rent-sc)/price*100
            label = f'{int(beds)}BR {loc.replace("Al Ranim ","AR")}'
            print(f"  {label:18s} AED {price:>10,.0f} AED {rent:>8,.0f} AED {sc:>6,.0f} {gross:>6.2f}% {net:>6.2f}% {min(len(ts),len(cs)):>5}")
            yield_rows.append({'Segment': label, 'Gross': gross, 'Net': net})

# Chart
if yield_rows:
    ydf = pd.DataFrame(yield_rows)
    x = np.arange(len(ydf))
    w = 0.35
    bars1 = axes[0].bar(x - w/2, ydf['Gross'], w, label='Gross Yield', color='#2BA84A', edgecolor='white')
    bars2 = axes[0].bar(x + w/2, ydf['Net'], w, label='Net Yield', color='#E63946', edgecolor='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ydf['Segment'], rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Yield %')
    axes[0].set_title('Gross vs Net Yield by Segment', fontweight='bold')
    axes[0].legend()

# Yield bar ranked
if yield_rows:
    ydf_sorted = ydf.sort_values('Net', ascending=True)
    colors = ['#2BA84A' if v > ydf['Net'].median() else '#F4A261' for v in ydf_sorted['Net']]
    bars3 = axes[1].barh(range(len(ydf_sorted)), ydf_sorted['Net'], color=colors, edgecolor='white')
    axes[1].set_yticks(range(len(ydf_sorted)))
    axes[1].set_yticklabels(ydf_sorted['Segment'], fontsize=9)
    axes[1].set_xlabel('Net Yield %')
    axes[1].set_title('Segments Ranked by Net Yield', fontweight='bold')
    axes[1].bar_label(bars3, [f'{v:.2f}%' for v in ydf_sorted['Net']], padding=5, fontsize=9)

plt.tight_layout()
plt.savefig('../notebooks/fig_09_yields.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# # Section 6: Price Trends — Where Is the Market Going?
#
# Using 12 months of transaction data, we measure the **direction and speed** of price movement.

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Overall price trend
price_monthly = trans.groupby('year_month').agg(
    median_price=('total_sales_price_val', 'median'),
    median_psf=('sales_price_sqft_unit', 'median'),
    count=('id', 'count'),
).reset_index().sort_values('year_month')

if len(price_monthly) > 2:
    axes[0,0].plot(price_monthly['year_month'], price_monthly['median_price']/1e6, 'o-',
                    color='#2E86AB', linewidth=2, markersize=8)
    x = np.arange(len(price_monthly))
    sl, ic, rv, _, _ = linregress(x, price_monthly['median_price'])
    axes[0,0].plot(price_monthly['year_month'], [(sl*i+ic)/1e6 for i in x],
                    '--', color='#E63946', linewidth=1.5)
    ann_pct = sl*12/price_monthly['median_price'].mean()*100
    axes[0,0].set_title(f'Median Price Trend ({ann_pct:+.1f}%/year)', fontweight='bold')
    axes[0,0].set_ylabel('AED Millions')
    plt.setp(axes[0,0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# PSF trend
if len(price_monthly) > 2:
    axes[0,1].plot(price_monthly['year_month'], price_monthly['median_psf'], 'o-',
                    color='#2BA84A', linewidth=2, markersize=8)
    sl2, ic2, _, _, _ = linregress(x, price_monthly['median_psf'])
    axes[0,1].plot(price_monthly['year_month'], [sl2*i+ic2 for i in x],
                    '--', color='#E63946', linewidth=1.5)
    ann_psf_pct = sl2*12/price_monthly['median_psf'].mean()*100
    axes[0,1].set_title(f'PSF Trend ({ann_psf_pct:+.1f}%/year)', fontweight='bold')
    axes[0,1].set_ylabel('AED / sqft')
    plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Volume trend
if len(price_monthly) > 1:
    axes[1,0].bar(price_monthly['year_month'], price_monthly['count'],
                   color='#2E86AB', edgecolor='white', alpha=0.7)
    axes[1,0].set_title('Monthly Transaction Volume', fontweight='bold')
    axes[1,0].set_ylabel('Count')
    plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Rent trend
if 'year_month' in contracts.columns:
    rent_monthly = contracts.groupby('year_month')['annualised_rental_price'].median().reset_index().sort_values('year_month')
    if len(rent_monthly) > 2:
        axes[1,1].plot(rent_monthly['year_month'], rent_monthly['annualised_rental_price']/1000, 'o-',
                        color='#1B3A5C', linewidth=2, markersize=8)
        x_r = np.arange(len(rent_monthly))
        sl_r, ic_r, _, _, _ = linregress(x_r, rent_monthly['annualised_rental_price'])
        ann_rent_pct = sl_r*12/rent_monthly['annualised_rental_price'].mean()*100
        axes[1,1].set_title(f'Rent Trend ({ann_rent_pct:+.1f}%/year)', fontweight='bold')
        axes[1,1].set_ylabel('AED Thousands / Year')
        plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.suptitle('Market Trends — 12-Month History', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('../notebooks/fig_09_trends.png', dpi=150, bbox_inches='tight')
plt.show()

if len(price_monthly) > 2:
    print(f"📈 TREND SUMMARY:")
    print(f"   Price: {ann_pct:+.1f}% annualized")
    print(f"   PSF:   {ann_psf_pct:+.1f}% annualized")
    if len(rent_monthly) > 2:
        print(f"   Rent:  {ann_rent_pct:+.1f}% annualized")

# %% [markdown]
# ---
# # Section 7: Total Cost of Ownership
#
# **Before investing, you must understand ALL costs — not just the price tag.**
#
# Here is a complete cost breakdown for a typical purchase:

# %%
print("=" * 65)
print("TOTAL COST OF OWNERSHIP — WORKED EXAMPLES")
print("=" * 65)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for i, beds in enumerate([3.0, 4.0]):
    t = trans[trans['beds_numeric'] == beds]
    c = contracts[contracts['beds_numeric'] == beds]
    price = t['total_sales_price_val'].median()
    size = t['unit_size_sqft'].median()
    rent = c['annualised_rental_price'].median() if len(c) > 0 else 0

    dld = price * DLD_FEE_PCT
    agent = price * AGENT_COMMISSION_PCT
    total_acq = price + dld + agent
    sc = size * SERVICE_CHARGE_PER_SQFT
    maint = price * MAINTENANCE_PCT
    net_income = rent - sc - maint

    print(f"\n{'='*55}")
    print(f"  {int(beds)}BR TOWNHOUSE — INVESTMENT BREAKDOWN")
    print(f"{'='*55}")
    print(f"  PURCHASE:")
    print(f"    Property price:      AED {price:>12,.0f}")
    print(f"    DLD transfer (4%):   AED {dld:>12,.0f}")
    print(f"    Agent fee (2%):      AED {agent:>12,.0f}")
    print(f"    TOTAL TO INVEST:     AED {total_acq:>12,.0f}")
    print(f"")
    print(f"  ANNUAL INCOME:")
    print(f"    Gross rent:          AED {rent:>12,.0f}")
    print(f"    Service charge:     -AED {sc:>12,.0f}  ({size:,.0f} sqft × 3.1)")
    print(f"    Maintenance (~1%):  -AED {maint:>12,.0f}")
    print(f"    NET INCOME:          AED {net_income:>12,.0f}")
    print(f"")
    print(f"  RETURNS:")
    print(f"    Gross yield:         {rent/price*100:.2f}%")
    print(f"    Net yield on price:  {net_income/price*100:.2f}%")
    print(f"    Net yield on total:  {net_income/total_acq*100:.2f}%")
    print(f"    Monthly cash flow:   AED {net_income/12:,.0f}")

    # Waterfall chart
    items = ['Purchase\nPrice', 'DLD\n(4%)', 'Agent\n(2%)', 'Total\nInvested',
             'Gross\nRent', 'Service\nCharge', 'Maint.\n(1%)', 'Net\nIncome']
    values = [price, dld, agent, total_acq, rent, -sc, -maint, net_income]
    colors_bar = ['#2E86AB', '#F4A261', '#9B59B6', '#1B3A5C', '#2BA84A', '#E63946', '#E63946', '#2BA84A']

    bars = axes[i].bar(items, values, color=colors_bar, edgecolor='white')
    axes[i].set_title(f'{int(beds)}BR — Cost & Income Breakdown', fontweight='bold')
    axes[i].yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
    axes[i].axhline(0, color='black', linewidth=0.5)
    for bar, val in zip(bars, values):
        axes[i].text(bar.get_x() + bar.get_width()/2, val,
                    f'AED {abs(val)/1000:,.0f}K', ha='center',
                    va='bottom' if val > 0 else 'top', fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig('../notebooks/fig_09_cost_breakdown.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# # Section 8: 5-Year Return Projections
#
# We model three scenarios based on different market conditions:
# - **Bear**: Rents flat, prices grow 3%/year
# - **Base**: Rents grow 5%/year, prices grow 8%/year
# - **Bull**: Rents grow 10%/year, prices grow 15%/year

# %%
base_price = trans['total_sales_price_val'].median()
base_rent = contracts['annualised_rental_price'].median() if len(contracts) > 0 else 185000
base_size = trans['unit_size_sqft'].median()
base_sc = base_size * SERVICE_CHARGE_PER_SQFT
base_maint = base_price * MAINTENANCE_PCT
acq_cost = base_price * (DLD_FEE_PCT + AGENT_COMMISSION_PCT)
total_invested = base_price + acq_cost

scenarios = {
    'Bear':  {'rent_g': 0.00, 'price_g': 0.03, 'color': '#E63946', 'marker': 'v'},
    'Base':  {'rent_g': 0.05, 'price_g': 0.08, 'color': '#2E86AB', 'marker': 'o'},
    'Bull':  {'rent_g': 0.10, 'price_g': 0.15, 'color': '#2BA84A', 'marker': '^'},
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
years = list(range(1, 6))

print("=" * 75)
print("5-YEAR RETURN PROJECTIONS")
print(f"Base: Purchase AED {base_price:,.0f} | Rent AED {base_rent:,.0f}/yr | Invested AED {total_invested:,.0f}")
print("=" * 75)

for name, p in scenarios.items():
    cumulative_roi = []
    annual_income = []
    for yr in years:
        # Cumulative net rental income
        net_inc = sum(base_rent*(1+p['rent_g'])**i - base_sc - base_maint for i in range(1, yr+1))
        # Capital gain
        cap = base_price * ((1+p['price_g'])**yr - 1)
        # Total return minus acquisition cost
        total = net_inc + cap - acq_cost
        roi = total / total_invested * 100
        cumulative_roi.append(roi)
        annual_income.append(base_rent*(1+p['rent_g'])**yr - base_sc - base_maint)

    axes[0].plot(years, cumulative_roi, f'{p["marker"]}-', linewidth=2.5, markersize=8,
                 label=f'{name} (rent +{p["rent_g"]*100:.0f}%, price +{p["price_g"]*100:.0f}%)',
                 color=p['color'])

    print(f"\n{name} scenario (rent +{p['rent_g']*100:.0f}%, price +{p['price_g']*100:.0f}%/yr):")
    for yr, roi, inc in zip(years, cumulative_roi, annual_income):
        print(f"  Year {yr}: {roi:>+7.1f}% ROI | Net income AED {inc:>10,.0f}/yr")

axes[0].set_xlabel('Year', fontsize=12)
axes[0].set_ylabel('Cumulative ROI %', fontsize=12)
axes[0].set_title('5-Year Investment Return Scenarios', fontweight='bold', fontsize=14)
axes[0].legend(fontsize=9)
axes[0].axhline(0, color='black', linewidth=0.5)
axes[0].grid(True, alpha=0.3)

# Break-even analysis
for name, p in scenarios.items():
    for yr in range(1, 21):
        net_inc = sum(base_rent*(1+p['rent_g'])**i - base_sc - base_maint for i in range(1, yr+1))
        cap = base_price * ((1+p['price_g'])**yr - 1)
        if net_inc + cap >= acq_cost:
            print(f"\n{name}: Break-even at year {yr} (acquisition costs recovered)")
            break

# Year 5 comparison bar chart
yr5_returns = {}
for name, p in scenarios.items():
    net_inc = sum(base_rent*(1+p['rent_g'])**i - base_sc - base_maint for i in range(1, 6))
    cap = base_price * ((1+p['price_g'])**5 - 1)
    yr5_returns[name] = {'Rental Income': net_inc, 'Capital Gain': cap, 'Acq. Cost': -acq_cost}

names = list(yr5_returns.keys())
x = np.arange(len(names))
w = 0.25
axes[1].bar(x-w, [yr5_returns[n]['Rental Income']/1e6 for n in names], w,
            label='Net Rental Income', color='#2BA84A', edgecolor='white')
axes[1].bar(x, [yr5_returns[n]['Capital Gain']/1e6 for n in names], w,
            label='Capital Gain', color='#2E86AB', edgecolor='white')
axes[1].bar(x+w, [yr5_returns[n]['Acq. Cost']/1e6 for n in names], w,
            label='Acquisition Cost', color='#E63946', edgecolor='white')
axes[1].set_xticks(x)
axes[1].set_xticklabels(names)
axes[1].set_ylabel('AED Millions')
axes[1].set_title('5-Year Return Components', fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('../notebooks/fig_09_5yr_returns.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# # Section 9: Investment Recommendations
#
# Based on our analysis of 404 actual transactions, 265 rental contracts, 15 ML models,
# and 12 months of market data, here are our recommendations:

# %%
print("=" * 85)
print("INVESTMENT RECOMMENDATIONS")
print("=" * 85)

print("""
┌─────────────────────────────────────────────────────────────────────────────────┐
│  INVESTOR PROFILE     │  RECOMMENDED STRATEGY                                  │
├───────────────────────┼────────────────────────────────────────────────────────│
│                       │                                                        │
│  CONSERVATIVE         │  3BR G+1 in Al Ranim 1 or 2                           │
│  (Capital protection) │  Entry: ~AED 3.0-3.4M | Net yield: ~5.5%             │
│                       │  Why: Established area, Title Deed transactions,       │
│                       │  proven rental demand, lowest risk                     │
│                       │                                                        │
│  BALANCED             │  4BR G+1 in Al Ranim 1 or 2                           │
│  (Yield + growth)     │  Entry: ~AED 3.8-4.2M | Net yield: ~5.7%             │
│                       │  Why: Higher rent, slightly better yield,              │
│                       │  strong tenant demand for family homes                 │
│                       │                                                        │
│  GROWTH-FOCUSED       │  3BR in Al Ranim 5-8 (Oqood stage)                    │
│  (Capital gains)      │  Entry: ~AED 3.0-3.3M | Net yield: TBD               │
│                       │  Why: Lowest PSF entry, newer units,                   │
│                       │  capital appreciation potential as area matures        │
│                       │                                                        │
│  YIELD MAXIMISER      │  3BR with proven rental in highest-yield sub-area      │
│  (Cash flow focus)    │  Entry: varies | Target net yield: >5.5%              │
│                       │  Why: Buy based purely on net yield ranking,           │
│                       │  secure tenant before closing if possible              │
│                       │                                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
""")

# %% [markdown]
# ---
# # Section 10: Risks & What to Watch Out For
#
# | Risk Factor | Impact | What To Do |
# |---|---|---|
# | **Listing prices are inflated** | You may overpay by ~3-10% | Always benchmark against our transaction data, not listings |
# | **Service charges may increase** | Currently 3.1 AED/sqft, could rise | Budget for 5-10% annual SC increases in your projections |
# | **Oqood areas (5-8) are unproven** | No rental track record yet | If buying here, accept higher risk for potential higher return |
# | **Limited 12-month history** | Trend projections have uncertainty | Use the confidence intervals, not just point estimates |
# | **Dubai market cycles** | Prices can correct | Focus on yield (rental income) not just capital appreciation |
# | **3BR vs 4BR break-even is 15+ years** | 4BR is NOT a yield play | Buy 4BR only if you believe in long-term capital appreciation |
# | **Vacancy risk** | Your unit may sit empty between tenants | Budget 1-2 months vacancy per year in worst case |
#
# ---
#
# # How to Use This Analysis
#
# 1. **Identify your budget** — this report shows median prices, but you can find deals above and below
# 2. **Check PSF** — always compare on a per-sqft basis, not headline price
# 3. **Verify any listing** — use our ML models (Notebook 02) to check if a listing is fairly priced
# 4. **Calculate YOUR yield** — plug in the actual asking price and estimated rent using our data
# 5. **Download everything** — all notebooks, data, and figures are in the repository for your use
#
# ---
#
# *This report was generated from actual DLD transaction data and registered rental contracts.
# All ML models were trained exclusively on verified transactions, not listing prices.
# Service charge of 3.1 AED/sqft is factored into all net yield calculations.*
