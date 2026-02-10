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
# # 💰 Investment Analysis — Synthesis
# ## Al-Ranim, Mudon, Dubai
#
# This is the **core investor decision notebook**. It synthesizes all prior analysis into:
# - Total cost of ownership with service charges
# - Net yield analysis across all segments
# - Location premium cost-benefit (park-facing, corner, single-row)
# - Sub-area ranking with composite investment score
# - Anomaly detection for undervalued properties
# - Head-to-head comparisons (3BR vs 4BR, G+1 vs G+2)
# - 1, 3, 5-year return scenarios

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

from sklearn.ensemble import IsolationForest

from src.config import *
from src.data_loader import load_sales_data, load_rental_data, get_transactions_only, get_contracts_only
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
listings_sales = sales[sales['data_source'] == 'listing']

bcol_r = 'beds' if 'beds' in contracts.columns else 'no_beds'
if 'beds_numeric' not in contracts.columns:
    contracts = contracts.copy()
    contracts['beds_numeric'] = pd.to_numeric(contracts[bcol_r], errors='coerce')

print(f"📊 {len(transactions)} sale transactions | {len(contracts)} rental contracts | {len(listings_sales)} sale listings")

# %% [markdown]
# ---
# ## 1. Total Cost of Ownership
#
# Investors must understand ALL costs, not just the headline price.

# %%
print("=" * 65)
print("TOTAL COST OF OWNERSHIP — AL RANIM TOWNHOUSE")
print("=" * 65)

for beds in sorted(transactions['beds_numeric'].dropna().unique()):
    sub = transactions[transactions['beds_numeric'] == beds]
    med_price = sub['total_sales_price_val'].median()
    med_size = sub['unit_size_sqft'].median()
    dld = med_price * DLD_FEE_PCT
    agent = med_price * AGENT_COMMISSION_PCT
    total_acq = med_price + dld + agent
    sc = med_size * SERVICE_CHARGE_PER_SQFT
    maint = med_price * MAINTENANCE_PCT

    print(f"\n{'='*50}")
    print(f"📐 {int(beds)}BR TOWNHOUSE")
    print(f"{'='*50}")
    print(f"  Purchase price (median):     AED {med_price:>12,.0f}")
    print(f"  DLD fee (4%):                AED {dld:>12,.0f}")
    print(f"  Agent commission (2%):       AED {agent:>12,.0f}")
    print(f"  ──────────────────────────────────────────")
    print(f"  Total acquisition cost:      AED {total_acq:>12,.0f}")
    print(f"")
    print(f"  Annual service charge:       AED {sc:>12,.0f}  ({med_size:,.0f} sqft × 3.1)")
    print(f"  Annual maintenance (~1%):    AED {maint:>12,.0f}")
    print(f"  Total annual expenses:       AED {sc + maint:>12,.0f}")

# %%
# Cost breakdown visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, beds in enumerate(sorted(transactions['beds_numeric'].dropna().unique())[:2]):
    sub = transactions[transactions['beds_numeric'] == beds]
    med_price = sub['total_sales_price_val'].median()
    med_size = sub['unit_size_sqft'].median()
    costs = {
        'Purchase Price': med_price,
        'DLD (4%)': med_price * DLD_FEE_PCT,
        'Agent (2%)': med_price * AGENT_COMMISSION_PCT,
    }
    annual = {
        'Service Charge': med_size * SERVICE_CHARGE_PER_SQFT,
        'Maintenance (~1%)': med_price * MAINTENANCE_PCT,
    }
    axes[i].bar(costs.keys(), costs.values(), color=['#2E86AB', '#F4A261', '#9B59B6'], edgecolor='white')
    axes[i].set_title(f'{int(beds)}BR — Acquisition Costs', fontweight='bold')
    axes[i].yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
    for j, (k, v) in enumerate(costs.items()):
        axes[i].text(j, v, f'AED {v:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('../notebooks/fig_06_acquisition_costs.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 2. Net Yield Analysis (Service Charge Adjusted)

# %%
yield_data = compute_yield_metrics(transactions, contracts, group_cols=['sub_loc_2', 'beds_numeric'])

if len(yield_data) > 0:
    print("=" * 80)
    print("NET YIELD BY SEGMENT (Actual Data Only)")
    print("=" * 80)
    yield_display = yield_data.copy()
    yield_display['median_price'] = yield_display['median_price'].apply(lambda x: f'AED {x:,.0f}')
    yield_display['median_rent'] = yield_display['median_rent'].apply(lambda x: f'AED {x:,.0f}')
    yield_display['annual_service_charge'] = yield_display['annual_service_charge'].apply(lambda x: f'AED {x:,.0f}')
    yield_display['gross_yield_pct'] = yield_display['gross_yield_pct'].round(2)
    yield_display['net_yield_pct'] = yield_display['net_yield_pct'].round(2)
    cols = ['sub_loc_2', 'beds_numeric', 'median_price', 'median_rent',
            'annual_service_charge', 'gross_yield_pct', 'net_yield_pct', 'price_count', 'rent_count']
    avail = [c for c in cols if c in yield_display.columns]
    print(yield_display[avail].to_string(index=False))

# %%
if len(yield_data) > 0:
    fig, ax = plt.subplots(figsize=(14, 6))
    segments = [f"{r['sub_loc_2']}\n{int(r['beds_numeric'])}BR" for _, r in yield_data.iterrows()]
    x = np.arange(len(segments))
    w = 0.35
    bars1 = ax.bar(x - w/2, yield_data['gross_yield_pct'], w, label='Gross Yield', color='#2BA84A', edgecolor='white')
    bars2 = ax.bar(x + w/2, yield_data['net_yield_pct'], w, label='Net Yield (after SC)', color='#E63946', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(segments, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Yield %')
    ax.set_title('Gross vs Net Yield by Segment', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.bar_label(bars1, fmt='%.1f%%', padding=2, fontsize=8)
    ax.bar_label(bars2, fmt='%.1f%%', padding=2, fontsize=8)
    plt.tight_layout()
    plt.savefig('../notebooks/fig_06_yield_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ---
# ## 3. Location Premium Cost-Benefit Analysis
#
# **Is paying extra for park-facing worth it?**

# %%
print("=" * 70)
print("LOCATION PREMIUM COST-BENEFIT")
print("=" * 70)

location_analysis = []
for feat, label in [('is_park_facing', 'Park Facing'), ('is_corner_unit', 'Corner Unit'),
                     ('is_single_row', 'Single Row')]:
    # Sale premium
    pf_sales = transactions[transactions[feat] == 1]['total_sales_price_val']
    npf_sales = transactions[transactions[feat] == 0]['total_sales_price_val']
    # Rent premium
    pf_rent = contracts[contracts[feat] == 1]['annualised_rental_price']
    npf_rent = contracts[contracts[feat] == 0]['annualised_rental_price']

    if len(pf_sales) > 3 and len(npf_sales) > 3:
        price_prem = pf_sales.median() - npf_sales.median()
        price_prem_pct = price_prem / npf_sales.median() * 100

        rent_prem = 0
        rent_prem_pct = 0
        if len(pf_rent) > 3 and len(npf_rent) > 3:
            rent_prem = pf_rent.median() - npf_rent.median()
            rent_prem_pct = rent_prem / npf_rent.median() * 100

        # Is it worth it? Extra rent / extra cost
        roi_premium = (rent_prem / price_prem * 100) if price_prem > 0 else 0

        location_analysis.append({
            'Feature': label,
            'Price Premium AED': price_prem,
            'Price Premium %': price_prem_pct,
            'Rent Premium AED': rent_prem,
            'Rent Premium %': rent_prem_pct,
            'Premium Yield (extra rent / extra cost)': roi_premium,
            'Sales Count': len(pf_sales),
            'Rent Count': len(pf_rent),
        })

        print(f"\n{label} ({len(pf_sales)} sales, {len(pf_rent)} rentals):")
        print(f"  Price premium:  AED {price_prem:>+10,.0f} ({price_prem_pct:+.1f}%)")
        print(f"  Rent premium:   AED {rent_prem:>+10,.0f}/yr ({rent_prem_pct:+.1f}%)")
        print(f"  Premium yield:  {roi_premium:.1f}% (extra rent as % of extra cost)")
        verdict = "✅ WORTH IT" if roi_premium > 4 else "⚠️ MARGINAL" if roi_premium > 2 else "❌ NOT WORTH IT"
        print(f"  Verdict:        {verdict}")
    else:
        print(f"\n{label}: Insufficient data")

# %%
if location_analysis:
    loc_df = pd.DataFrame(location_analysis)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Price vs Rent premium
    x = np.arange(len(loc_df))
    w = 0.35
    axes[0].bar(x - w/2, loc_df['Price Premium %'], w, label='Price Premium %', color='#E63946', edgecolor='white')
    axes[0].bar(x + w/2, loc_df['Rent Premium %'], w, label='Rent Premium %', color='#2BA84A', edgecolor='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(loc_df['Feature'])
    axes[0].set_ylabel('Premium %')
    axes[0].set_title('Price vs Rent Premium by Location Feature', fontweight='bold')
    axes[0].legend()

    # Premium yield
    colors = ['#2BA84A' if v > 4 else '#F4A261' if v > 2 else '#E63946' for v in loc_df['Premium Yield (extra rent / extra cost)']]
    bars = axes[1].bar(loc_df['Feature'], loc_df['Premium Yield (extra rent / extra cost)'], color=colors, edgecolor='white')
    axes[1].set_ylabel('Premium Yield %')
    axes[1].set_title('Is the Location Premium Worth It?', fontweight='bold')
    axes[1].axhline(4, color='#2BA84A', linestyle='--', alpha=0.5, label='Good threshold (4%)')
    axes[1].legend()
    axes[1].bar_label(bars, fmt='%.1f%%', padding=3)

    plt.tight_layout()
    plt.savefig('../notebooks/fig_06_location_costbenefit.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ---
# ## 4. Anomaly Detection — Undervalued Properties

# %%
# Isolation Forest on sale listings
feat_cols = ['total_sales_price_val', 'sales_price_sqft_unit', 'unit_size_sqft',
             'beds_numeric', 'is_g_plus_2', 'is_park_facing', 'is_corner_unit', 'is_single_row']

list_df = listings_sales.dropna(subset=['total_sales_price_val', 'unit_size_sqft', 'sales_price_sqft_unit']).copy()
list_df['beds_numeric'] = list_df['beds_numeric'].fillna(3)
list_df['is_g_plus_2'] = list_df['is_g_plus_2'].fillna(0)
X_list = list_df[feat_cols].fillna(0)

iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
list_df['anomaly_score'] = iso.fit_predict(X_list)
list_df['anomaly_score_raw'] = iso.decision_function(X_list)

# Also Z-score within segment
list_df['z_score'] = 0.0
for beds in list_df['beds_numeric'].unique():
    mask = list_df['beds_numeric'] == beds
    sub = list_df.loc[mask, 'sales_price_sqft_unit']
    if sub.std() > 0:
        list_df.loc[mask, 'z_score'] = (sub - sub.mean()) / sub.std()

# Undervalued = anomaly + low price (negative z-score)
list_df['potentially_undervalued'] = ((list_df['anomaly_score'] == -1) & (list_df['z_score'] < -0.5)).astype(int)

n_anomalies = (list_df['anomaly_score'] == -1).sum()
n_undervalued = list_df['potentially_undervalued'].sum()

print(f"Isolation Forest: {n_anomalies} anomalies detected ({n_anomalies/len(list_df)*100:.1f}%)")
print(f"Potentially undervalued (anomaly + low Z-score): {n_undervalued}")

# %%
fig, ax = plt.subplots(figsize=(12, 7))
normal = list_df[list_df['anomaly_score'] == 1]
anomaly = list_df[list_df['anomaly_score'] == -1]
underval = list_df[list_df['potentially_undervalued'] == 1]

ax.scatter(normal['unit_size_sqft'], normal['total_sales_price_val'], alpha=0.2, s=15, color='#2E86AB', label='Normal Listings')
ax.scatter(anomaly['unit_size_sqft'], anomaly['total_sales_price_val'], alpha=0.6, s=40, color='#F4A261', marker='s', label=f'Anomalies ({n_anomalies})')
if len(underval) > 0:
    ax.scatter(underval['unit_size_sqft'], underval['total_sales_price_val'], alpha=0.8, s=80, color='#2BA84A', marker='*', label=f'Potentially Undervalued ({n_undervalued})')
ax.set_xlabel('Unit Size (sqft)')
ax.set_ylabel('Listing Price (AED)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
ax.set_title('Anomaly Detection — Undervalued Property Opportunities', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('../notebooks/fig_06_anomaly_detection.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
if n_undervalued > 0:
    print("\n🟢 TOP POTENTIALLY UNDERVALUED LISTINGS:")
    display_cols = ['sub_loc_2', 'no_beds', 'unit_size_sqft', 'total_sales_price_val',
                    'sales_price_sqft_unit', 'is_park_facing', 'z_score']
    avail = [c for c in display_cols if c in underval.columns]
    print(underval.nsmallest(10, 'z_score')[avail].to_string(index=False))

# %% [markdown]
# ---
# ## 5. Sub-Area Comparison Matrix (Al Ranim 1-8)

# %%
subloc_scores = []
for loc in SUBLOC_ORDER:
    t = transactions[transactions['sub_loc_2'] == loc]
    c = contracts[contracts['sub_loc_2'] == loc]
    l = listings_sales[listings_sales['sub_loc_2'] == loc]

    if len(t) < 3:
        continue

    score = {
        'Sub-Area': loc,
        'Median Price (AED)': t['total_sales_price_val'].median(),
        'Median PSF (AED)': t['sales_price_sqft_unit'].median(),
        'Transaction Count': len(t),
        'Listing Count': len(l),
        '% Park Facing': t['is_park_facing'].mean() * 100,
    }

    if len(c) > 2:
        med_rent = c['annualised_rental_price'].median()
        med_price = t['total_sales_price_val'].median()
        med_size = t['unit_size_sqft'].median()
        sc = med_size * SERVICE_CHARGE_PER_SQFT
        score['Median Rent (AED)'] = med_rent
        score['Gross Yield %'] = (med_rent / med_price) * 100
        score['Net Yield %'] = ((med_rent - sc) / med_price) * 100
    else:
        score['Median Rent (AED)'] = np.nan
        score['Gross Yield %'] = np.nan
        score['Net Yield %'] = np.nan

    # Price volatility (CV)
    score['Price CV %'] = (t['total_sales_price_val'].std() / t['total_sales_price_val'].mean()) * 100

    subloc_scores.append(score)

subloc_df = pd.DataFrame(subloc_scores)

print("=" * 90)
print("SUB-AREA COMPARISON MATRIX")
print("=" * 90)
print(subloc_df.round(2).to_string(index=False))

# %%
# Composite investment score
if len(subloc_df) > 0:
    score_df = subloc_df.copy()
    # Normalize each metric to 0-10
    def norm_score(series, higher_better=True):
        if series.max() == series.min():
            return pd.Series([5]*len(series))
        normed = (series - series.min()) / (series.max() - series.min()) * 10
        return normed if higher_better else 10 - normed

    score_df['yield_score'] = norm_score(score_df['Net Yield %'].fillna(0))
    score_df['liquidity_score'] = norm_score(score_df['Transaction Count'])
    score_df['value_score'] = norm_score(score_df['Median PSF (AED)'], higher_better=False)  # Lower PSF = better value
    score_df['risk_score'] = norm_score(score_df['Price CV %'], higher_better=False)  # Lower CV = lower risk

    # Weighted composite
    score_df['Composite Score'] = (
        score_df['yield_score'] * 0.30 +
        score_df['liquidity_score'] * 0.20 +
        score_df['value_score'] * 0.25 +
        score_df['risk_score'] * 0.25
    )
    score_df = score_df.sort_values('Composite Score', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [PALETTE.get(loc, '#999') for loc in score_df['Sub-Area']]
    bars = ax.barh(score_df['Sub-Area'], score_df['Composite Score'], color=colors, edgecolor='white')
    ax.set_xlabel('Composite Investment Score (0-10)')
    ax.set_title('Sub-Area Investment Ranking', fontweight='bold', fontsize=14)
    ax.bar_label(bars, fmt='%.1f', padding=5, fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('../notebooks/fig_06_subloc_ranking.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n🏆 SUB-AREA INVESTMENT RANKING:")
    for i, (_, row) in enumerate(score_df.iterrows()):
        medal = ['🥇', '🥈', '🥉'][i] if i < 3 else f' {i+1}.'
        print(f"  {medal} {row['Sub-Area']:15s} — Score: {row['Composite Score']:.1f}/10")

# %% [markdown]
# ---
# ## 6. Head-to-Head: 3BR vs 4BR

# %%
print("=" * 60)
print("HEAD-TO-HEAD: 3BR vs 4BR TOWNHOUSE")
print("=" * 60)

for beds in [3.0, 4.0]:
    t = transactions[transactions['beds_numeric'] == beds]
    c = contracts[contracts['beds_numeric'] == beds]
    print(f"\n{int(beds)}BR:")
    print(f"  Median price:        AED {t['total_sales_price_val'].median():,.0f}")
    print(f"  Median size:         {t['unit_size_sqft'].median():,.0f} sqft")
    print(f"  Transactions:        {len(t)}")
    if len(c) > 0:
        med_rent = c['annualised_rental_price'].median()
        sc = t['unit_size_sqft'].median() * SERVICE_CHARGE_PER_SQFT
        print(f"  Median annual rent:  AED {med_rent:,.0f}")
        print(f"  Service charge:      AED {sc:,.0f}")
        gross = (med_rent / t['total_sales_price_val'].median()) * 100
        net = ((med_rent - sc) / t['total_sales_price_val'].median()) * 100
        print(f"  Gross yield:         {gross:.2f}%")
        print(f"  Net yield:           {net:.2f}%")

# Break-even
t3 = transactions[transactions['beds_numeric'] == 3]
t4 = transactions[transactions['beds_numeric'] == 4]
c3 = contracts[contracts['beds_numeric'] == 3]
c4 = contracts[contracts['beds_numeric'] == 4]

if len(t3) > 0 and len(t4) > 0 and len(c3) > 0 and len(c4) > 0:
    extra_cost = t4['total_sales_price_val'].median() - t3['total_sales_price_val'].median()
    extra_rent = c4['annualised_rental_price'].median() - c3['annualised_rental_price'].median()
    if extra_rent > 0:
        breakeven_years = extra_cost / extra_rent
        print(f"\n📊 4BR vs 3BR: Extra cost AED {extra_cost:,.0f}, extra rent AED {extra_rent:,.0f}/yr")
        print(f"   Break-even: {breakeven_years:.1f} years to recoup extra cost via higher rent")

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics_3 = []
metrics_4 = []
labels = ['Median Price', 'Median Rent', 'Net Yield %']

for beds, metrics in [(3.0, metrics_3), (4.0, metrics_4)]:
    t = transactions[transactions['beds_numeric'] == beds]
    c = contracts[contracts['beds_numeric'] == beds]
    metrics.append(t['total_sales_price_val'].median())
    metrics.append(c['annualised_rental_price'].median() if len(c) > 0 else 0)
    sc = t['unit_size_sqft'].median() * SERVICE_CHARGE_PER_SQFT
    rent = c['annualised_rental_price'].median() if len(c) > 0 else 0
    net = ((rent - sc) / t['total_sales_price_val'].median()) * 100 if t['total_sales_price_val'].median() > 0 else 0
    metrics.append(net)

for ax, label, v3, v4 in zip(axes, labels, metrics_3, metrics_4):
    bars = ax.bar(['3BR', '4BR'], [v3, v4], color=['#2E86AB', '#E63946'], edgecolor='white', width=0.5)
    ax.set_title(label, fontweight='bold')
    if 'Price' in label or 'Rent' in label:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
        ax.bar_label(bars, [f'AED {v:,.0f}' for v in [v3, v4]], padding=3, fontsize=9)
    else:
        ax.bar_label(bars, [f'{v:.2f}%' for v in [v3, v4]], padding=3, fontsize=9)

plt.suptitle('3BR vs 4BR — Head-to-Head Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../notebooks/fig_06_3br_vs_4br.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 7. Long-Term Scenarios (1, 3, 5 Year)

# %%
print("=" * 75)
print("LONG-TERM INVESTMENT SCENARIOS")
print("=" * 75)

base_price = transactions['total_sales_price_val'].median()
base_rent = contracts['annualised_rental_price'].median() if len(contracts) > 0 else 180000
base_size = transactions['unit_size_sqft'].median()
base_sc = base_size * SERVICE_CHARGE_PER_SQFT
acq_cost = base_price * (DLD_FEE_PCT + AGENT_COMMISSION_PCT)

print(f"Base: Price AED {base_price:,.0f} | Rent AED {base_rent:,.0f}/yr | SC AED {base_sc:,.0f}/yr | Acq AED {acq_cost:,.0f}")

scenarios = {
    'Bear':  {'rent_g': 0.00, 'price_g': 0.03, 'color': '#E63946'},
    'Base':  {'rent_g': 0.05, 'price_g': 0.08, 'color': '#2E86AB'},
    'Bull':  {'rent_g': 0.10, 'price_g': 0.15, 'color': '#2BA84A'},
}

fig, ax = plt.subplots(figsize=(12, 6))
years = list(range(1, 6))

for name, p in scenarios.items():
    cumulative = []
    for yr in years:
        net_income = sum(base_rent * (1+p['rent_g'])**i - base_sc for i in range(1, yr+1))
        cap_gain = base_price * ((1+p['price_g'])**yr - 1)
        total = ((net_income + cap_gain - acq_cost) / (base_price + acq_cost)) * 100
        cumulative.append(total)

    ax.plot(years, cumulative, 'o-', linewidth=2.5, markersize=8, label=name, color=p['color'])

    print(f"\n{name} (rent +{p['rent_g']*100:.0f}%/yr, price +{p['price_g']*100:.0f}%/yr):")
    for yr, ret in zip(years, cumulative):
        print(f"  Year {yr}: {ret:+.1f}% total return")

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Cumulative Total Return %', fontsize=12)
ax.set_title('5-Year Investment Return Scenarios', fontweight='bold', fontsize=14)
ax.legend(fontsize=11)
ax.axhline(0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../notebooks/fig_06_5yr_scenarios.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## Key Takeaways for Investors
#
# 1. **Acquisition costs add 6%** on top of purchase price (DLD + agent). Factor this into all ROI calculations.
#
# 2. **Service charge (3.1 AED/sqft)** reduces gross yield by a measurable amount.
#    All yield figures in this analysis are presented net of service charges.
#
# 3. **Location premiums**: The data quantifies exactly whether park-facing, corner,
#    or single-row properties justify their premium through higher rents.
#
# 4. **Sub-area ranking**: The composite score identifies which Al Ranim sub-areas
#    offer the best risk-adjusted investment opportunity.
#
# 5. **3BR vs 4BR**: The break-even analysis shows how long it takes for the extra
#    rental income from 4BR to justify the higher purchase price.
#
# 6. **5-year outlook**: Even in the Bear scenario, the investment generates positive
#    returns over the medium term, supported by rental income and moderate capital appreciation.
