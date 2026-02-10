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
# # 📐 Price Per Square Foot — Deep Analysis
# ## Al-Ranim, Mudon, Dubai
#
# Price per square foot (PSF) is the **single most important metric** for comparing
# properties fairly. A 3BR at AED 3.2M and a 4BR at AED 4.0M look very different —
# but when you compare on a per-sqft basis, the picture changes completely.
#
# This notebook answers:
# - What is the current PSF across Al-Ranim?
# - How does PSF differ by unit type, bedrooms, floor level, and sub-area?
# - Where is PSF trending over the next 12 months?
# - Which unit types offer the best value per sqft?
# - How does rent per sqft compare, and what does that mean for yields?

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
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.config import *
from src.data_loader import load_sales_data, load_rental_data, get_transactions_only, get_contracts_only, get_listings_only
from src.feature_engineering import extract_comment_features, normalize_developer_type, create_sales_features, create_rental_features
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

# Ensure rent_per_sqft exists
if 'rent_per_sqft' not in contracts.columns:
    contracts['rent_per_sqft'] = contracts['annualised_rental_price'] / contracts[scol]

print(f"✅ Loaded: {len(trans)} transactions | {len(contracts)} contracts | {len(listings)} listings")

# %% [markdown]
# ---
# ## 1. Overall PSF Distribution
#
# What does the price per square foot landscape look like across ALL actual transactions?

# %%
psf = trans['sales_price_sqft_unit'].dropna()

print("=" * 60)
print("PRICE PER SQUARE FOOT — OVERALL (Transactions Only)")
print("=" * 60)
print(f"  Mean:    AED {psf.mean():,.0f} /sqft")
print(f"  Median:  AED {psf.median():,.0f} /sqft")
print(f"  Min:     AED {psf.min():,.0f} /sqft")
print(f"  Max:     AED {psf.max():,.0f} /sqft")
print(f"  Std:     AED {psf.std():,.0f} /sqft")
print(f"  Range:   AED {psf.max() - psf.min():,.0f} /sqft spread")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Histogram
axes[0].hist(psf, bins=40, color='#2E86AB', edgecolor='white', alpha=0.8)
axes[0].axvline(psf.median(), color='#E63946', linewidth=2, linestyle='--',
                label=f'Median: AED {psf.median():,.0f}')
axes[0].axvline(psf.mean(), color='#F4A261', linewidth=2, linestyle=':',
                label=f'Mean: AED {psf.mean():,.0f}')
axes[0].set_xlabel('Price per Sqft (AED)')
axes[0].set_ylabel('Count')
axes[0].set_title('PSF Distribution — All Transactions', fontweight='bold')
axes[0].legend(fontsize=10)

# Box plot by evidence type
existing_evd = [e for e in ['Title Deed', 'Oqood', 'Active Listings'] if e in sales['evdnc_name'].values]
colors_evd = [PALETTE.get(e, '#999') for e in existing_evd]
data_groups = [sales[sales['evdnc_name']==e]['sales_price_sqft_unit'].dropna() for e in existing_evd]
bp = axes[1].boxplot(data_groups, labels=existing_evd, patch_artist=True)
for patch, color in zip(bp['boxes'], colors_evd):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_ylabel('Price per Sqft (AED)')
axes[1].set_title('PSF by Record Type — Transactions vs Listings', fontweight='bold')
for i, grp in enumerate(data_groups):
    axes[1].text(i+1, grp.median(), f'AED {grp.median():,.0f}', ha='center', va='bottom',
                fontweight='bold', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

plt.tight_layout()
plt.savefig('../notebooks/fig_08_psf_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 2. PSF by Bedrooms & Floor Level
#
# Different unit configurations have fundamentally different PSF economics.

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# By bedrooms
for beds in sorted(trans['beds_numeric'].dropna().unique()):
    sub = trans[trans['beds_numeric'] == beds]
    med_psf = sub['sales_price_sqft_unit'].median()
    med_size = sub['unit_size_sqft'].median()
    print(f"{int(beds)}BR: Median PSF AED {med_psf:,.0f} | Size {med_size:,.0f} sqft | {len(sub)} transactions")

beds_data = trans.groupby('beds_numeric')['sales_price_sqft_unit'].median()
bars = axes[0].bar([f'{int(b)}BR' for b in beds_data.index], beds_data.values,
                   color=['#2E86AB', '#E63946'], edgecolor='white', width=0.5)
axes[0].set_title('Median PSF by Bedrooms', fontweight='bold')
axes[0].set_ylabel('AED / sqft')
axes[0].bar_label(bars, [f'AED {v:,.0f}' for v in beds_data.values], padding=5, fontsize=10, fontweight='bold')

# By floor level
fl_data = trans.groupby('floor_level')['sales_price_sqft_unit'].median()
bars2 = axes[1].bar(fl_data.index, fl_data.values, color=['#2BA84A', '#9B59B6'], edgecolor='white', width=0.5)
axes[1].set_title('Median PSF by Floor Level', fontweight='bold')
axes[1].set_ylabel('AED / sqft')
axes[1].bar_label(bars2, [f'AED {v:,.0f}' for v in fl_data.values], padding=5, fontsize=10, fontweight='bold')

# By beds × floor
cross = trans.groupby(['beds_numeric', 'floor_level'])['sales_price_sqft_unit'].median().unstack()
cross.plot(kind='bar', ax=axes[2], color=['#2BA84A', '#9B59B6'], edgecolor='white')
axes[2].set_title('Median PSF by Beds × Floor Level', fontweight='bold')
axes[2].set_ylabel('AED / sqft')
axes[2].set_xticklabels([f'{int(b)}BR' for b in cross.index], rotation=0)
axes[2].legend(title='Floor')

plt.tight_layout()
plt.savefig('../notebooks/fig_08_psf_by_config.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 3. PSF by Sub-Area
#
# This reveals which sub-areas are **cheaper per sqft** (better value entry points)
# and which command a premium.

# %%
subloc_psf = trans.groupby('sub_loc_2').agg(
    median_psf=('sales_price_sqft_unit', 'median'),
    mean_psf=('sales_price_sqft_unit', 'mean'),
    count=('sales_price_sqft_unit', 'count'),
    std_psf=('sales_price_sqft_unit', 'std'),
).reindex([s for s in SUBLOC_ORDER if s in trans['sub_loc_2'].values])

print("=" * 70)
print("PSF BY SUB-AREA (Transactions Only)")
print("=" * 70)
for loc, row in subloc_psf.iterrows():
    print(f"  {loc:15s}: AED {row['median_psf']:>7,.0f}/sqft (±{row['std_psf']:,.0f}) | {int(row['count'])} trans")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
colors = [PALETTE.get(loc, '#999') for loc in subloc_psf.index]
bars = axes[0].bar(range(len(subloc_psf)), subloc_psf['median_psf'], color=colors, edgecolor='white')
axes[0].set_xticks(range(len(subloc_psf)))
axes[0].set_xticklabels(subloc_psf.index, rotation=45, ha='right')
axes[0].set_ylabel('Median PSF (AED)')
axes[0].set_title('Median Price per Sqft by Sub-Area', fontweight='bold')
axes[0].bar_label(bars, [f'{v:,.0f}' for v in subloc_psf['median_psf']], padding=3, fontsize=9)

# Box plot
subloc_order_valid = [s for s in SUBLOC_ORDER if s in trans['sub_loc_2'].values]
bp_data = [trans[trans['sub_loc_2']==loc]['sales_price_sqft_unit'].dropna() for loc in subloc_order_valid]
bp = axes[1].boxplot(bp_data, labels=[s.replace('Al Ranim ', 'AR') for s in subloc_order_valid],
                      patch_artist=True)
for patch, loc in zip(bp['boxes'], subloc_order_valid):
    patch.set_facecolor(PALETTE.get(loc, '#999'))
    patch.set_alpha(0.7)
axes[1].set_ylabel('Price per Sqft (AED)')
axes[1].set_title('PSF Distribution by Sub-Area', fontweight='bold')

plt.tight_layout()
plt.savefig('../notebooks/fig_08_psf_by_subloc.png', dpi=150, bbox_inches='tight')
plt.show()

# Find cheapest and most expensive
cheapest = subloc_psf['median_psf'].idxmin()
priciest = subloc_psf['median_psf'].idxmax()
spread = subloc_psf['median_psf'].max() - subloc_psf['median_psf'].min()
print(f"\n📊 Cheapest PSF: {cheapest} (AED {subloc_psf.loc[cheapest, 'median_psf']:,.0f}/sqft)")
print(f"📊 Priciest PSF: {priciest} (AED {subloc_psf.loc[priciest, 'median_psf']:,.0f}/sqft)")
print(f"📊 Spread: AED {spread:,.0f}/sqft between cheapest and priciest area")

# %% [markdown]
# ---
# ## 4. PSF by Unit Type / Developer Type
#
# Different configurations (3B1, 3B2, 4B1, 4B2) have different layouts, sizes, and PSF.

# %%
# By unit subtype (extracted from developer_type)
if 'unit_subtype' in trans.columns:
    subtype_psf = trans.groupby('unit_subtype').agg(
        median_psf=('sales_price_sqft_unit', 'median'),
        median_price=('total_sales_price_val', 'median'),
        median_size=('unit_size_sqft', 'median'),
        count=('id', 'count'),
    ).dropna()

    if len(subtype_psf) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        print("\n📐 PSF BY UNIT SUBTYPE:")
        for st, row in subtype_psf.iterrows():
            print(f"  {st:6s}: PSF AED {row['median_psf']:>7,.0f} | Price AED {row['median_price']:>10,.0f} | "
                  f"Size {row['median_size']:>6,.0f} sqft | {int(row['count'])} trans")

        bars = axes[0].bar(subtype_psf.index, subtype_psf['median_psf'],
                           color=sns.color_palette('husl', len(subtype_psf)), edgecolor='white')
        axes[0].set_title('Median PSF by Unit Subtype', fontweight='bold')
        axes[0].set_ylabel('AED / sqft')
        axes[0].bar_label(bars, [f'AED {v:,.0f}' for v in subtype_psf['median_psf']], padding=3, fontsize=9)

        # Size vs PSF scatter
        for st in trans['unit_subtype'].dropna().unique():
            sub = trans[trans['unit_subtype'] == st]
            axes[1].scatter(sub['unit_size_sqft'], sub['sales_price_sqft_unit'],
                           alpha=0.4, s=30, label=st)
        axes[1].set_xlabel('Unit Size (sqft)')
        axes[1].set_ylabel('Price per Sqft (AED)')
        axes[1].set_title('PSF vs Unit Size by Subtype', fontweight='bold')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig('../notebooks/fig_08_psf_by_subtype.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print("Limited subtype data — skipping visualization")
else:
    print("No unit_subtype column — skipping")

# %% [markdown]
# ---
# ## 5. PSF by Location Features (Park-Facing, Corner, Single Row)

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

location_features = [('is_park_facing', 'Park Facing'),
                     ('is_corner_unit', 'Corner Unit'),
                     ('is_single_row', 'Single Row')]

print("=" * 60)
print("PSF BY LOCATION FEATURE")
print("=" * 60)

for ax, (feat, label) in zip(axes, location_features):
    yes = trans[trans[feat] == 1]['sales_price_sqft_unit']
    no = trans[trans[feat] == 0]['sales_price_sqft_unit']

    if len(yes) > 3:
        premium = yes.median() - no.median()
        premium_pct = (premium / no.median()) * 100
        print(f"\n{label}:")
        print(f"  Yes: AED {yes.median():,.0f}/sqft ({len(yes)} trans)")
        print(f"  No:  AED {no.median():,.0f}/sqft ({len(no)} trans)")
        print(f"  PSF premium: AED {premium:+,.0f}/sqft ({premium_pct:+.1f}%)")

        bp_data = [no, yes]
        bp = ax.boxplot(bp_data, labels=[f'No\n(n={len(no)})', f'Yes\n(n={len(yes)})'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#F4A261')
        bp['boxes'][1].set_facecolor('#2BA84A')
        for b in bp['boxes']:
            b.set_alpha(0.7)
        ax.set_title(f'PSF: {label}', fontweight='bold')
        ax.set_ylabel('AED / sqft')
    else:
        ax.text(0.5, 0.5, f'{label}\nInsufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'PSF: {label}', fontweight='bold')

plt.tight_layout()
plt.savefig('../notebooks/fig_08_psf_location.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 6. Rent Per Square Foot Analysis
#
# Rent PSF is equally important — it tells investors the income-earning efficiency per sqft owned.

# %%
rpsf = contracts['rent_per_sqft'].dropna()

print("=" * 60)
print("RENT PER SQUARE FOOT (Contracts Only)")
print("=" * 60)
print(f"  Median: AED {rpsf.median():,.1f} /sqft/year")
print(f"  Mean:   AED {rpsf.mean():,.1f} /sqft/year")
print(f"  Range:  AED {rpsf.min():,.1f} — AED {rpsf.max():,.1f} /sqft/year")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Distribution
axes[0].hist(rpsf, bins=30, color='#1B3A5C', edgecolor='white', alpha=0.8)
axes[0].axvline(rpsf.median(), color='#E63946', linewidth=2, linestyle='--',
                label=f'Median: AED {rpsf.median():,.1f}')
axes[0].set_xlabel('Rent per Sqft (AED/year)')
axes[0].set_title('Rent PSF Distribution', fontweight='bold')
axes[0].legend()

# By beds
for beds in sorted(contracts['beds_numeric'].dropna().unique()):
    sub = contracts[contracts['beds_numeric'] == beds]
    med = sub['rent_per_sqft'].median()
    print(f"\n{int(beds)}BR rent PSF: AED {med:,.1f}/sqft/year ({len(sub)} contracts)")

rent_beds = contracts.groupby('beds_numeric')['rent_per_sqft'].median()
bars = axes[1].bar([f'{int(b)}BR' for b in rent_beds.index], rent_beds.values,
                   color=['#2E86AB', '#E63946'], edgecolor='white', width=0.5)
axes[1].set_title('Rent PSF by Bedrooms', fontweight='bold')
axes[1].set_ylabel('AED / sqft / year')
axes[1].bar_label(bars, [f'AED {v:,.1f}' for v in rent_beds.values], padding=5, fontsize=10, fontweight='bold')

# By sub-area
subloc_rent_psf = contracts.groupby('sub_loc_2')['rent_per_sqft'].median()
subloc_rent_psf = subloc_rent_psf.reindex([s for s in SUBLOC_ORDER if s in subloc_rent_psf.index])
if len(subloc_rent_psf) > 0:
    colors = [PALETTE.get(loc, '#999') for loc in subloc_rent_psf.index]
    bars2 = axes[2].bar(range(len(subloc_rent_psf)), subloc_rent_psf.values, color=colors, edgecolor='white')
    axes[2].set_xticks(range(len(subloc_rent_psf)))
    axes[2].set_xticklabels([s.replace('Al Ranim ', 'AR') for s in subloc_rent_psf.index], rotation=45, ha='right')
    axes[2].set_title('Rent PSF by Sub-Area', fontweight='bold')
    axes[2].set_ylabel('AED / sqft / year')
    axes[2].bar_label(bars2, [f'{v:,.1f}' for v in subloc_rent_psf.values], padding=3, fontsize=9)

plt.tight_layout()
plt.savefig('../notebooks/fig_08_rent_psf.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 7. PSF Yield — The Complete Picture
#
# **PSF Yield = Rent per sqft / Price per sqft × 100**
# This is the purest yield metric — it strips out unit size entirely.

# %%
# Match at segment level
print("=" * 60)
print("PSF YIELD BY SEGMENT")
print("=" * 60)

psf_yield_data = []
for beds in [3.0, 4.0]:
    for loc in SUBLOC_ORDER:
        t = trans[(trans['beds_numeric']==beds) & (trans['sub_loc_2']==loc)]
        c = contracts[(contracts['beds_numeric']==beds) & (contracts['sub_loc_2']==loc)]
        if len(t) >= 5 and len(c) >= 5:
            sale_psf = t['sales_price_sqft_unit'].median()
            rent_psf = c['rent_per_sqft'].median()
            sc_psf = SERVICE_CHARGE_PER_SQFT
            gross_psf_yield = (rent_psf / sale_psf) * 100
            net_psf_yield = ((rent_psf - sc_psf) / sale_psf) * 100
            psf_yield_data.append({
                'Sub-Area': loc, 'Beds': int(beds),
                'Sale PSF': sale_psf, 'Rent PSF': rent_psf,
                'SC PSF': sc_psf,
                'Gross PSF Yield %': round(gross_psf_yield, 2),
                'Net PSF Yield %': round(net_psf_yield, 2),
                'Sales': len(t), 'Contracts': len(c),
            })

if psf_yield_data:
    psf_yield_df = pd.DataFrame(psf_yield_data)
    print(psf_yield_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(14, 6))
    segments = [f"{r['Sub-Area']}\n{r['Beds']}BR" for _, r in psf_yield_df.iterrows()]
    x = np.arange(len(segments))
    w = 0.35
    bars1 = ax.bar(x - w/2, psf_yield_df['Gross PSF Yield %'], w, label='Gross PSF Yield', color='#2BA84A', edgecolor='white')
    bars2 = ax.bar(x + w/2, psf_yield_df['Net PSF Yield %'], w, label='Net PSF Yield (after 3.1 SC)', color='#E63946', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(segments, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Yield %')
    ax.set_title('PSF Yield by Segment — Gross vs Net', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.bar_label(bars1, fmt='%.1f%%', padding=2, fontsize=8)
    ax.bar_label(bars2, fmt='%.1f%%', padding=2, fontsize=8)
    plt.tight_layout()
    plt.savefig('../notebooks/fig_08_psf_yield.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ---
# ## 8. PSF Trends Over Time — Where Is It Going?

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Overall PSF trend
trans_monthly = trans.groupby('year_month').agg(
    median_psf=('sales_price_sqft_unit', 'median'),
    mean_psf=('sales_price_sqft_unit', 'mean'),
    count=('id', 'count'),
).reset_index().sort_values('year_month')

if len(trans_monthly) > 2:
    axes[0,0].plot(trans_monthly['year_month'], trans_monthly['median_psf'], 'o-',
                    color='#2E86AB', linewidth=2, markersize=8)
    axes[0,0].set_title('Median PSF Trend — All Transactions', fontweight='bold')
    axes[0,0].set_ylabel('AED / sqft')
    plt.setp(axes[0,0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Linear trend
    x = np.arange(len(trans_monthly))
    slope, intercept, r_val, p_val, _ = linregress(x, trans_monthly['median_psf'])
    annual_change = slope * 12
    annual_pct = (annual_change / trans_monthly['median_psf'].mean()) * 100
    axes[0,0].plot(trans_monthly['year_month'],
                    [slope*i + intercept for i in x],
                    '--', color='#E63946', linewidth=1.5, label=f'Trend: {annual_pct:+.1f}%/yr')
    axes[0,0].legend()

    print(f"📈 PSF TREND: {annual_pct:+.1f}% annualized ({annual_change:+,.0f} AED/sqft/year)")

# PSF trend by bedrooms
for beds in sorted(trans['beds_numeric'].dropna().unique()):
    sub = trans[trans['beds_numeric'] == beds]
    monthly = sub.groupby('year_month')['sales_price_sqft_unit'].median().reset_index()
    monthly = monthly.sort_values('year_month')
    if len(monthly) > 2:
        axes[0,1].plot(monthly['year_month'], monthly['sales_price_sqft_unit'], 'o-',
                        linewidth=2, markersize=6, label=f'{int(beds)}BR')
axes[0,1].set_title('PSF Trend by Bedrooms', fontweight='bold')
axes[0,1].set_ylabel('AED / sqft')
axes[0,1].legend()
plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45, ha='right')

# PSF trend by sub-area
for loc in SUBLOC_ORDER:
    sub = trans[trans['sub_loc_2'] == loc]
    if len(sub) > 15:
        monthly = sub.groupby('year_month')['sales_price_sqft_unit'].median().reset_index()
        monthly = monthly.sort_values('year_month')
        if len(monthly) > 2:
            axes[1,0].plot(monthly['year_month'], monthly['sales_price_sqft_unit'], 'o-',
                            linewidth=2, markersize=5, label=loc.replace('Al Ranim ', 'AR'),
                            color=PALETTE.get(loc, '#999'))
axes[1,0].set_title('PSF Trend by Sub-Area', fontweight='bold')
axes[1,0].set_ylabel('AED / sqft')
axes[1,0].legend(fontsize=8)
plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Listing PSF vs Transaction PSF over time
list_monthly = listings.groupby('year_month')['sales_price_sqft_unit'].median().reset_index().sort_values('year_month')
if len(trans_monthly) > 1:
    axes[1,1].plot(trans_monthly['year_month'], trans_monthly['median_psf'], 'o-',
                    color='#2BA84A', linewidth=2, markersize=6, label='Transaction PSF')
if len(list_monthly) > 1:
    axes[1,1].plot(list_monthly['year_month'], list_monthly['sales_price_sqft_unit'], 's--',
                    color='#F4A261', linewidth=2, markersize=5, label='Listing PSF')
axes[1,1].set_title('Transaction PSF vs Listing PSF Over Time', fontweight='bold')
axes[1,1].set_ylabel('AED / sqft')
axes[1,1].legend()
plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../notebooks/fig_08_psf_trends.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 9. 12-Month PSF Projection

# %%
if len(trans_monthly) > 3:
    ts = trans_monthly.set_index('year_month')['median_psf'].sort_index()
    ts.index = pd.RangeIndex(len(ts))
    periods = 12

    try:
        model = ExponentialSmoothing(ts.values, trend='add', seasonal=None)
        fit = model.fit(optimized=True)
        forecast = fit.forecast(periods)
    except Exception:
        x = np.arange(len(ts))
        sl, ic, _, _, _ = linregress(x, ts.values)
        forecast = sl * np.arange(len(ts), len(ts)+periods) + ic

    # Bootstrap CI
    n_boot = 200
    boot_fc = np.zeros((n_boot, periods))
    for b in range(n_boot):
        idx = np.sort(np.random.choice(len(ts), len(ts), replace=True))
        bts = ts.values[idx]
        try:
            bm = ExponentialSmoothing(bts, trend='add', seasonal=None)
            bf = bm.fit(optimized=True)
            boot_fc[b] = bf.forecast(periods)
        except Exception:
            xb = np.arange(len(bts))
            sl2, ic2, _, _, _ = linregress(xb, bts)
            boot_fc[b] = sl2 * np.arange(len(bts), len(bts)+periods) + ic2

    ci_lo = np.percentile(boot_fc, 10, axis=0)
    ci_hi = np.percentile(boot_fc, 90, axis=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    months_hist = trans_monthly['year_month'].tolist()
    months_future = [f'M+{i+1}' for i in range(periods)]

    ax.plot(range(len(ts)), ts.values, 'o-', color='#2E86AB', linewidth=2, markersize=6, label='Historical PSF')
    ax.plot(range(len(ts), len(ts)+periods), forecast, 's--', color='#E63946', linewidth=2, markersize=6, label='12-Month Projection')
    ax.fill_between(range(len(ts), len(ts)+periods), ci_lo, ci_hi,
                     alpha=0.2, color='#E63946', label='80% Confidence Band')
    ax.axvline(len(ts)-0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xticks(range(len(months_hist)+len(months_future)))
    ax.set_xticklabels(months_hist + months_future, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Median PSF (AED/sqft)')
    ax.set_title('12-Month Price Per Sqft Projection', fontweight='bold', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig('../notebooks/fig_08_psf_projection.png', dpi=150, bbox_inches='tight')
    plt.show()

    pct_change = (forecast[-1] - ts.values[-1]) / ts.values[-1] * 100
    print(f"\n📈 12-MONTH PSF PROJECTION:")
    print(f"   Current: AED {ts.values[-1]:,.0f}/sqft")
    print(f"   12-month: AED {forecast[-1]:,.0f}/sqft ({pct_change:+.1f}%)")
    print(f"   80% CI: AED {ci_lo[-1]:,.0f} — AED {ci_hi[-1]:,.0f}/sqft")

# %% [markdown]
# ---
# ## 10. PSF Value Map — Where To Buy
#
# Combining PSF data across all dimensions to identify the **best value zones**.

# %%
print("=" * 75)
print("PSF VALUE MAP — BEST VALUE ZONES")
print("=" * 75)

value_data = []
for beds in [3.0, 4.0]:
    for loc in SUBLOC_ORDER:
        t = trans[(trans['beds_numeric']==beds) & (trans['sub_loc_2']==loc)]
        if len(t) >= 3:
            med_psf = t['sales_price_sqft_unit'].median()
            med_price = t['total_sales_price_val'].median()
            med_size = t['unit_size_sqft'].median()
            value_data.append({
                'Area': loc, 'Beds': int(beds),
                'PSF (AED)': med_psf, 'Price (AED)': med_price,
                'Size (sqft)': med_size, 'Count': len(t),
            })

value_df = pd.DataFrame(value_data).sort_values('PSF (AED)')
print(value_df.to_string(index=False))

if len(value_df) > 0:
    fig, ax = plt.subplots(figsize=(14, 7))

    for beds in [3, 4]:
        sub = value_df[value_df['Beds'] == beds]
        color = '#2E86AB' if beds == 3 else '#E63946'
        ax.scatter(sub['Size (sqft)'], sub['PSF (AED)'], s=sub['Count']*8,
                   alpha=0.7, color=color, edgecolors='white', linewidth=1,
                   label=f'{beds}BR')
        for _, row in sub.iterrows():
            ax.annotate(row['Area'].replace('Al Ranim ', 'AR'),
                       (row['Size (sqft)'], row['PSF (AED)']),
                       textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel('Median Unit Size (sqft)')
    ax.set_ylabel('Median Price per Sqft (AED)')
    ax.set_title('PSF Value Map: Size vs Price Efficiency\n(Bubble size = transaction count)', fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('../notebooks/fig_08_psf_value_map.png', dpi=150, bbox_inches='tight')
    plt.show()

    cheapest_entry = value_df.iloc[0]
    print(f"\n🏆 BEST VALUE (lowest PSF): {cheapest_entry['Area']} {cheapest_entry['Beds']}BR")
    print(f"   AED {cheapest_entry['PSF (AED)']:,.0f}/sqft | Total AED {cheapest_entry['Price (AED)']:,.0f}")

# %% [markdown]
# ---
# ## Key Takeaways — Price Per Square Foot
#
# 1. **PSF is the great equalizer** — comparing a 3BR at AED 3.2M to a 4BR at AED 4.0M
#    is misleading. On a per-sqft basis, the difference tells a different story.
#
# 2. **Sub-area matters more than bedrooms** for PSF — the cheapest area can be
#    significantly cheaper per sqft than the priciest.
#
# 3. **PSF is trending** — the 12-month projection shows the direction of the market
#    on a normalized basis.
#
# 4. **Rent PSF reveals true income efficiency** — some areas earn more rent per sqft
#    owned than others, directly impacting investor returns.
#
# 5. **The PSF Value Map** identifies exactly which sub-area × bedroom combinations
#    offer the best entry price per square foot.
