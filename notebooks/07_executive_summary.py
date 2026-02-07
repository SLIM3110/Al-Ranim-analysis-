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
# # 📋 Executive Summary — Al-Ranim Investment Report
# ## Mudon, Dubai | Prepared February 2026
#
# **This report synthesizes the complete analysis of Al-Ranim real estate market data,
# distilled into actionable insights for investors.**
#
# ---

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

transactions = get_transactions_only(sales)
contracts = get_contracts_only(rental)
listings = get_listings_only(sales)

bcol_r = 'beds' if 'beds' in contracts.columns else 'no_beds'
if 'beds_numeric' not in contracts.columns:
    contracts = contracts.copy()
    contracts['beds_numeric'] = pd.to_numeric(contracts[bcol_r], errors='coerce')

# %% [markdown]
# ---
# ## Market Overview

# %%
fig, ax = plt.subplots(figsize=(14, 3))
ax.axis('off')

stats = [
    ('Total Records', f'{len(sales):,} sales\n{len(rental):,} rental'),
    ('Actual Transactions', f'{len(transactions):,} sales\n{len(contracts):,} rental'),
    ('Price Range', f'AED {transactions["total_sales_price_val"].min():,.0f}\nto AED {transactions["total_sales_price_val"].max():,.0f}'),
    ('Rent Range', f'AED {contracts["annualised_rental_price"].min():,.0f}\nto AED {contracts["annualised_rental_price"].max():,.0f}'),
    ('Property Types', f'Townhouses & Villas\n3BR & 4BR'),
    ('Developer', 'Dubai Properties\nMudon Community'),
]

for i, (title, value) in enumerate(stats):
    x = i / len(stats)
    ax.text(x + 0.5/len(stats), 0.75, title, transform=ax.transAxes,
            fontsize=11, ha='center', fontweight='bold', color='#1B3A5C')
    ax.text(x + 0.5/len(stats), 0.25, value, transform=ax.transAxes,
            fontsize=10, ha='center', color='#333333')

plt.tight_layout()
plt.savefig('../notebooks/fig_07_overview.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## Key Finding 1: Listing vs Transaction Price Gap
#
# **Listing prices are consistently higher than actual transaction prices.**
# Investors should negotiate based on transaction data, not asking prices.

# %%
trans_med = transactions['total_sales_price_val'].median()
list_med = listings['total_sales_price_val'].median()
gap_pct = (list_med - trans_med) / trans_med * 100

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(['Transaction\nMedian', 'Listing\nMedian'], [trans_med, list_med],
              color=['#2BA84A', '#F4A261'], edgecolor='white', width=0.5)
ax.bar_label(bars, [f'AED {trans_med:,.0f}', f'AED {list_med:,.0f}'], padding=5, fontsize=11, fontweight='bold')
ax.set_title(f'Listing Premium: {gap_pct:+.1f}% Above Transaction Prices', fontweight='bold', fontsize=13)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
plt.tight_layout()
plt.savefig('../notebooks/fig_07_listing_gap.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## Key Finding 2: Net Yield After Service Charges

# %%
yield_data = compute_yield_metrics(transactions, contracts, group_cols=['sub_loc_2', 'beds_numeric'])

if len(yield_data) > 0:
    avg_gross = yield_data['gross_yield_pct'].mean()
    avg_net = yield_data['net_yield_pct'].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(['Gross Yield', 'Net Yield\n(after 3.1 AED/sqft SC)'],
                  [avg_gross, avg_net], color=['#2BA84A', '#E63946'], edgecolor='white', width=0.5)
    ax.bar_label(bars, [f'{avg_gross:.2f}%', f'{avg_net:.2f}%'], padding=5, fontsize=12, fontweight='bold')
    ax.set_title('Average Yield: Gross vs Net (Service Charge Impact)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Yield %')
    plt.tight_layout()
    plt.savefig('../notebooks/fig_07_yield_summary.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"📊 Service charge: 3.1 AED/sqft reduces yield by ~{avg_gross - avg_net:.2f} percentage points")

# %% [markdown]
# ---
# ## Key Finding 3: Location Premium Impact

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Park facing price
for pf, label, color in [(0, 'Not Park Facing', '#F4A261'), (1, 'Park Facing', '#2BA84A')]:
    sub = transactions[transactions['is_park_facing'] == pf]
    axes[0].hist(sub['total_sales_price_val'].dropna(), bins=25, alpha=0.5, color=color,
                 label=f'{label} (n={len(sub)}, med=AED {sub["total_sales_price_val"].median():,.0f})', edgecolor='white')
axes[0].set_title('Sale Price: Park Facing Impact', fontweight='bold')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
axes[0].legend(fontsize=9)

# Park facing rent
for pf, label, color in [(0, 'Not Park Facing', '#F4A261'), (1, 'Park Facing', '#2BA84A')]:
    sub = contracts[contracts['is_park_facing'] == pf]
    if len(sub) > 0:
        axes[1].hist(sub['annualised_rental_price'].dropna(), bins=20, alpha=0.5, color=color,
                     label=f'{label} (n={len(sub)}, med=AED {sub["annualised_rental_price"].median():,.0f})', edgecolor='white')
axes[1].set_title('Annual Rent: Park Facing Impact', fontweight='bold')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('../notebooks/fig_07_park_facing.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## Key Finding 4: Sub-Area Investment Ranking

# %%
subloc_data = []
for loc in SUBLOC_ORDER:
    t = transactions[transactions['sub_loc_2'] == loc]
    c = contracts[contracts['sub_loc_2'] == loc]
    if len(t) >= 3:
        entry = {'Area': loc, 'Trans.': len(t), 'Med Price': t['total_sales_price_val'].median(),
                 'Med PSF': t['sales_price_sqft_unit'].median()}
        if len(c) >= 3:
            rent = c['annualised_rental_price'].median()
            sc = t['unit_size_sqft'].median() * SERVICE_CHARGE_PER_SQFT
            entry['Med Rent'] = rent
            entry['Net Yield'] = ((rent - sc) / t['total_sales_price_val'].median()) * 100
        else:
            entry['Med Rent'] = np.nan
            entry['Net Yield'] = np.nan
        subloc_data.append(entry)

subloc_summary = pd.DataFrame(subloc_data)

if len(subloc_summary) > 0:
    fig, ax = plt.subplots(figsize=(12, 5))
    valid = subloc_summary.dropna(subset=['Net Yield'])
    if len(valid) > 0:
        colors = [PALETTE.get(a, '#999') for a in valid['Area']]
        bars = ax.bar(valid['Area'], valid['Net Yield'], color=colors, edgecolor='white')
        ax.set_ylabel('Net Yield %')
        ax.set_title('Net Yield by Sub-Area (After Service Charges)', fontweight='bold', fontsize=13)
        ax.bar_label(bars, fmt='%.2f%%', padding=3, fontsize=10, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../notebooks/fig_07_subloc_yield.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n📊 Sub-Area Summary:")
    print(subloc_summary.to_string(index=False))

# %% [markdown]
# ---
# ## Key Finding 5: 3BR vs 4BR Investment Verdict

# %%
comparison = []
for beds in [3.0, 4.0]:
    t = transactions[transactions['beds_numeric'] == beds]
    c = contracts[contracts['beds_numeric'] == beds]
    entry = {'Beds': f'{int(beds)}BR', 'Med Price': t['total_sales_price_val'].median(),
             'Med Size': t['unit_size_sqft'].median(), 'Count': len(t)}
    if len(c) > 0:
        entry['Med Rent'] = c['annualised_rental_price'].median()
        sc = t['unit_size_sqft'].median() * SERVICE_CHARGE_PER_SQFT
        entry['Service Charge'] = sc
        entry['Net Yield'] = ((entry['Med Rent'] - sc) / entry['Med Price']) * 100
    comparison.append(entry)

comp_df = pd.DataFrame(comparison)
print(comp_df.to_string(index=False))

# %% [markdown]
# ---
# ## Investment Recommendation Matrix

# %%
print("=" * 85)
print("INVESTMENT RECOMMENDATION MATRIX")
print("=" * 85)

recs = [
    {"Profile": "Conservative",   "Segment": "3BR G+1 Al Ranim 1-3 (Title Deed area)",
     "Entry": "~AED 3.0-3.5M", "Risk": "Low",    "Outlook": "Stable yields, proven area"},
    {"Profile": "Balanced",       "Segment": "4BR G+1 Al Ranim 1-3",
     "Entry": "~AED 3.8-4.5M", "Risk": "Medium", "Outlook": "Higher rent, good appreciation"},
    {"Profile": "Growth-Focused", "Segment": "3BR Al Ranim 5-8 (Oqood stage)",
     "Entry": "~AED 2.8-3.3M", "Risk": "Higher", "Outlook": "Lower entry, capital growth potential"},
    {"Profile": "Yield-Focused",  "Segment": "Park-facing 3BR with established tenants",
     "Entry": "~AED 3.3-3.8M", "Risk": "Medium", "Outlook": "Premium rent, stable demand"},
]
recs_df = pd.DataFrame(recs)
print(recs_df.to_string(index=False))

# %% [markdown]
# ---
# ## Worked Example: Buying a 3BR Townhouse in Al Ranim

# %%
price = transactions[transactions['beds_numeric'] == 3]['total_sales_price_val'].median()
size = transactions[transactions['beds_numeric'] == 3]['unit_size_sqft'].median()
rent = contracts[contracts['beds_numeric'] == 3]['annualised_rental_price'].median() if len(contracts[contracts['beds_numeric'] == 3]) > 0 else 190000
sc = size * SERVICE_CHARGE_PER_SQFT
dld = price * DLD_FEE_PCT
agent = price * AGENT_COMMISSION_PCT
maint = price * MAINTENANCE_PCT
total_acq = price + dld + agent

print("=" * 55)
print("WORKED EXAMPLE: 3BR TOWNHOUSE INVESTMENT")
print("=" * 55)
print(f"\n📐 Unit size: {size:,.0f} sqft")
print(f"\nACQUISITION:")
print(f"  Purchase price:         AED {price:>10,.0f}")
print(f"  DLD fee (4%):           AED {dld:>10,.0f}")
print(f"  Agent (2%):             AED {agent:>10,.0f}")
print(f"  Total cost:             AED {total_acq:>10,.0f}")
print(f"\nANNUAL INCOME & EXPENSES:")
print(f"  Gross annual rent:      AED {rent:>10,.0f}")
print(f"  Service charge:        -AED {sc:>10,.0f}")
print(f"  Maintenance (~1%):     -AED {maint:>10,.0f}")
print(f"  Net annual income:      AED {rent - sc - maint:>10,.0f}")
print(f"\nYIELDS:")
print(f"  Gross yield:            {rent/price*100:.2f}%")
print(f"  Net yield (on price):   {(rent-sc-maint)/price*100:.2f}%")
print(f"  Net yield (on total):   {(rent-sc-maint)/total_acq*100:.2f}%")

# 5-year projection
print(f"\n5-YEAR PROJECTION (Base scenario: rent +5%/yr, price +8%/yr):")
total_income = sum(rent * 1.05**i - sc - maint for i in range(1, 6))
cap_gain = price * (1.08**5 - 1)
total_return = total_income + cap_gain - (dld + agent)
roi = total_return / total_acq * 100
print(f"  Cumulative net income:  AED {total_income:>10,.0f}")
print(f"  Capital appreciation:   AED {cap_gain:>10,.0f}")
print(f"  Less acquisition fees: -AED {dld+agent:>10,.0f}")
print(f"  Total 5yr return:       AED {total_return:>10,.0f}")
print(f"  5yr ROI:                {roi:.1f}%")
print(f"  Annualised ROI:         {(1+roi/100)**(1/5)*100-100:.1f}%")

# %% [markdown]
# ---
# ## Risk Factors & Caveats
#
# | Risk | Impact | Mitigation |
# |------|--------|------------|
# | **Limited transaction data** (~400 sales, ~265 rentals) | Model accuracy limited | Use conservative estimates, cross-validate |
# | **12 months of history only** | Trend projections uncertain | Wide confidence intervals, scenario analysis |
# | **Service charge may increase** | Net yield reduction | Currently 3.1 AED/sqft — budget for 5-10% annual increase |
# | **Oqood-stage areas (5-8)** | Execution & delivery risk | Higher potential return but less proven |
# | **Dubai market cyclicality** | Capital value risk | Focus on yield; don't rely solely on appreciation |
# | **Listing price inflation** | Overpaying risk | Always benchmark against transaction data, not listings |
# | **Maintenance costs** | Hidden expense | Budget 1% of property value annually |
# | **Vacancy risk** | Income interruption | Al Ranim occupancy appears strong but not guaranteed |
#
# ---
#
# ## How to Use This Report
#
# 1. **Notebook 01** — Understand the data quality and features available
# 2. **Notebook 02** — Use the ML price models to validate any specific property's asking price
# 3. **Notebook 03** — Use rent models to estimate realistic rental income; check yield calculations
# 4. **Notebook 04** — See which investment tier a property falls into
# 5. **Notebook 05** — Check the 12-month price and rent projections
# 6. **Notebook 06** — Detailed investment analysis, anomaly detection, sub-area rankings
# 7. **This notebook** — Share with stakeholders as the executive summary
#
# **All data, code, and figures can be downloaded and reproduced.**
