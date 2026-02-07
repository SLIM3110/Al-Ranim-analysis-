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
# # 🏠 Rent Prediction & Yield Modeling
# ## Al-Ranim, Mudon, Dubai
#
# This notebook mirrors the sale price prediction approach but for **rental prices**.
# We then combine sale + rent predictions to model **investment yields** —
# the metric investors care about most.
#
# **Three Parts:**
# 1. **Rent Prediction** — 4 ML models predicting annual rent
# 2. **Yield Mapping** — Net yield after 3.1 AED/sqft service charge
# 3. **Long-Term Outlook** — Scenario analysis for 1, 3, 5 year returns

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
import warnings, time
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import xgboost as xgb

from src.config import *
from src.data_loader import load_sales_data, load_rental_data, get_transactions_only, get_contracts_only, get_listings_only
from src.feature_engineering import (extract_comment_features, create_rental_features,
                                     create_sales_features, normalize_developer_type, compute_yield_metrics)
from src.visualization import setup_style, format_aed

setup_style()
print("✅ Libraries loaded")

# %% [markdown]
# ## Part A: Rent Prediction Models

# %%
rental = load_rental_data()
rental = extract_comment_features(rental)
rental = create_rental_features(rental)

contracts = get_contracts_only(rental)
rental_listings = get_listings_only(rental)

beds_col = 'beds' if 'beds' in rental.columns else 'no_beds'
size_col = 'unit_size' if 'unit_size' in rental.columns else 'unit_size_sqft'

print(f"📊 Training data (actual contracts): {len(contracts):,}")
print(f"📊 Prediction data (rental listings): {len(rental_listings):,}")

# %%
numeric_features = [size_col, 'plot_size', 'months_since_start']
categorical_features = ['sub_loc_2']
binary_features = ['beds_numeric', 'is_g_plus_2', 'is_park_facing', 'is_corner_unit',
                   'is_single_row', 'is_end_unit', 'is_road_facing']
if 'is_furnished' in contracts.columns:
    binary_features.append('is_furnished')

all_features = numeric_features + categorical_features + binary_features
target = 'annualised_rental_price'

df_train = contracts[all_features + [target]].copy()

# Impute missing plot_size
for beds in df_train['beds_numeric'].dropna().unique():
    mask = (df_train['beds_numeric'] == beds) & (df_train['plot_size'].isna())
    med = df_train.loc[df_train['beds_numeric'] == beds, 'plot_size'].median()
    if pd.isna(med):
        med = df_train['plot_size'].median()
    df_train.loc[mask, 'plot_size'] = med

# Fill remaining NaN in plot_size
df_train['plot_size'] = df_train['plot_size'].fillna(df_train['plot_size'].median())
df_train = df_train.dropna(subset=[target, size_col])

X = df_train[all_features]
y = df_train[target]

print(f"\n✅ Training set: {X.shape[0]} rows × {X.shape[1]} features")
print(f"   Annual rent range: AED {y.min():,.0f} — AED {y.max():,.0f}")
print(f"   Median: AED {y.median():,.0f}")

# %%
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='infrequent_if_exist'), categorical_features),
    ('bin', 'passthrough', binary_features),
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

# %% [markdown]
# ### Model 1: Ridge Regression — Rent Drivers

# %%
t0 = time.time()
ridge_pipe = Pipeline([('pre', preprocessor), ('model', Ridge())])
param_grid = {'model__alpha': [0.1, 1, 10, 100, 1000]}
ridge_cv = GridSearchCV(ridge_pipe, param_grid, cv=kf, scoring='r2', n_jobs=-1)
ridge_cv.fit(X, y)
best_ridge = ridge_cv.best_estimator_

r2 = cross_val_score(best_ridge, X, y, cv=kf, scoring='r2')
rmse = np.sqrt(-cross_val_score(best_ridge, X, y, cv=kf, scoring='neg_mean_squared_error'))
mae = -cross_val_score(best_ridge, X, y, cv=kf, scoring='neg_mean_absolute_error')
results['Ridge'] = {'R2': r2.mean(), 'RMSE': rmse.mean(), 'MAE': mae.mean(), 'Time': time.time()-t0}

print(f"Ridge (alpha={ridge_cv.best_params_['model__alpha']}): R²={r2.mean():.4f}, RMSE=AED {rmse.mean():,.0f}, MAE=AED {mae.mean():,.0f}")

# Extract coefficients
best_ridge.fit(X, y)
feat_names = [str(f).replace('num__','').replace('cat__','').replace('bin__','')
              for f in best_ridge.named_steps['pre'].get_feature_names_out()]
coefs = best_ridge.named_steps['model'].coef_
coef_df = pd.DataFrame({'Feature': feat_names, 'Coefficient': coefs}).sort_values('Coefficient')

fig, ax = plt.subplots(figsize=(12, max(5, len(coef_df)*0.35)))
colors = ['#2BA84A' if c > 0 else '#E63946' for c in coef_df['Coefficient']]
ax.barh(range(len(coef_df)), coef_df['Coefficient'], color=colors, edgecolor='white')
ax.set_yticks(range(len(coef_df)))
ax.set_yticklabels(coef_df['Feature'])
ax.set_title('Ridge — Rent Price Drivers (AED/year)', fontweight='bold')
ax.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('../notebooks/fig_03_ridge_rent_coefs.png', dpi=150, bbox_inches='tight')
plt.show()

# Location premium from Ridge
print("\n📍 LOCATION PREMIUM ON RENT:")
for feat in ['is_park_facing', 'is_corner_unit', 'is_single_row']:
    if feat in coef_df['Feature'].values:
        val = coef_df.loc[coef_df['Feature']==feat, 'Coefficient'].values[0]
        print(f"   {feat:25s} → AED {val:>+10,.0f}/year ({val/y.median()*100:+.1f}%)")

# %% [markdown]
# ### Model 2: Random Forest — Rent Feature Interactions

# %%
t0 = time.time()
rf_pipe = Pipeline([('pre', preprocessor), ('model', RandomForestRegressor(random_state=42, n_jobs=-1))])
rf_cv = GridSearchCV(rf_pipe, {'model__n_estimators': [100, 300], 'model__max_depth': [8, 12, None],
                                'model__min_samples_leaf': [3, 5]}, cv=kf, scoring='r2', n_jobs=-1)
rf_cv.fit(X, y)
best_rf = rf_cv.best_estimator_

r2 = cross_val_score(best_rf, X, y, cv=kf, scoring='r2')
rmse = np.sqrt(-cross_val_score(best_rf, X, y, cv=kf, scoring='neg_mean_squared_error'))
mae = -cross_val_score(best_rf, X, y, cv=kf, scoring='neg_mean_absolute_error')
results['Random Forest'] = {'R2': r2.mean(), 'RMSE': rmse.mean(), 'MAE': mae.mean(), 'Time': time.time()-t0}

print(f"Random Forest: R²={r2.mean():.4f}, RMSE=AED {rmse.mean():,.0f}, MAE=AED {mae.mean():,.0f}")

best_rf.fit(X, y)
importances = best_rf.named_steps['model'].feature_importances_
fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values('Importance')

fig, ax = plt.subplots(figsize=(10, max(5, len(fi_df)*0.35)))
ax.barh(range(len(fi_df)), fi_df['Importance'], color='#2E86AB', edgecolor='white')
ax.set_yticks(range(len(fi_df)))
ax.set_yticklabels(fi_df['Feature'])
ax.set_title('Random Forest — Rent Feature Importance', fontweight='bold')
plt.tight_layout()
plt.savefig('../notebooks/fig_03_rf_rent_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Model 3: XGBoost — Best Rent Accuracy

# %%
t0 = time.time()
X_proc = preprocessor.fit_transform(X)
xgb_model = xgb.XGBRegressor(learning_rate=0.05, n_estimators=500, max_depth=4,
                               subsample=0.8, random_state=42, n_jobs=-1)
r2 = cross_val_score(xgb_model, X_proc, y, cv=kf, scoring='r2')
rmse = np.sqrt(-cross_val_score(xgb_model, X_proc, y, cv=kf, scoring='neg_mean_squared_error'))
mae = -cross_val_score(xgb_model, X_proc, y, cv=kf, scoring='neg_mean_absolute_error')
results['XGBoost'] = {'R2': r2.mean(), 'RMSE': rmse.mean(), 'MAE': mae.mean(), 'Time': time.time()-t0}

print(f"XGBoost: R²={r2.mean():.4f}, RMSE=AED {rmse.mean():,.0f}, MAE=AED {mae.mean():,.0f}")

xgb_model.fit(X_proc, y)

try:
    import shap
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_proc)
    fig, ax = plt.subplots(figsize=(12, 7))
    shap.summary_plot(shap_vals, X_proc, feature_names=feat_names, show=False, max_display=12)
    plt.title('XGBoost — SHAP Rent Feature Importance', fontweight='bold')
    plt.tight_layout()
    plt.savefig('../notebooks/fig_03_shap_rent.png', dpi=150, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"SHAP skipped: {e}")

# %% [markdown]
# ### Model 4: KNN — Comparable Rentals

# %%
t0 = time.time()
k_range = range(3, 16)
k_scores = [cross_val_score(KNeighborsRegressor(n_neighbors=k), X_proc, y, cv=kf, scoring='r2').mean() for k in k_range]
best_k = list(k_range)[np.argmax(k_scores)]

knn = KNeighborsRegressor(n_neighbors=best_k)
r2 = cross_val_score(knn, X_proc, y, cv=kf, scoring='r2')
rmse = np.sqrt(-cross_val_score(knn, X_proc, y, cv=kf, scoring='neg_mean_squared_error'))
mae = -cross_val_score(knn, X_proc, y, cv=kf, scoring='neg_mean_absolute_error')
results['KNN'] = {'R2': r2.mean(), 'RMSE': rmse.mean(), 'MAE': mae.mean(), 'Time': time.time()-t0}

print(f"KNN (k={best_k}): R²={r2.mean():.4f}, RMSE=AED {rmse.mean():,.0f}, MAE=AED {mae.mean():,.0f}")

# %% [markdown]
# ### Model Comparison

# %%
results_df = pd.DataFrame(results).T
results_df['RMSE'] = results_df['RMSE'].round(0)
results_df['MAE'] = results_df['MAE'].round(0)
results_df['R2'] = results_df['R2'].round(4)
print("RENT PREDICTION MODEL COMPARISON")
print(results_df[['R2','RMSE','MAE']].to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
models = list(results.keys())
colors = sns.color_palette('husl', len(models))
for ax, metric, title in zip(axes, ['R2','RMSE','MAE'], ['R² (Higher=Better)','RMSE (Lower=Better)','MAE (Lower=Better)']):
    vals = [results[m][metric] for m in models]
    bars = ax.bar(range(len(models)), vals, color=colors, edgecolor='white')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.set_title(title, fontweight='bold')
    if metric in ['RMSE','MAE']:
        ax.bar_label(bars, [f'AED {v:,.0f}' for v in vals], padding=3, fontsize=8)
    else:
        ax.bar_label(bars, [f'{v:.3f}' for v in vals], padding=3, fontsize=9)
plt.suptitle('Rent Prediction — Model Comparison', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../notebooks/fig_03_rent_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## Part B: Yield Mapping (Net Yield After Service Charges)
#
# **Gross Yield** = Annual Rent / Purchase Price × 100
# **Net Yield** = (Annual Rent − Service Charge) / Purchase Price × 100
# Service Charge = Unit Size (sqft) × 3.1 AED

# %%
sales = load_sales_data()
sales = extract_comment_features(sales)
sales = normalize_developer_type(sales)
sales = create_sales_features(sales)

sale_transactions = get_transactions_only(sales)
rent_contracts = get_contracts_only(rental)

# Ensure beds_numeric in both
if 'beds_numeric' not in rent_contracts.columns:
    rent_contracts = rent_contracts.copy()
    bcol = 'beds' if 'beds' in rent_contracts.columns else 'no_beds'
    rent_contracts['beds_numeric'] = pd.to_numeric(rent_contracts[bcol], errors='coerce')

yield_data = compute_yield_metrics(sale_transactions, rent_contracts, group_cols=['sub_loc_2', 'beds_numeric'])

print("=" * 60)
print("YIELD BY SEGMENT (Actual Transactions & Contracts Only)")
print("=" * 60)
if len(yield_data) > 0:
    display_cols = ['sub_loc_2', 'beds_numeric', 'median_price', 'median_rent',
                    'annual_service_charge', 'gross_yield_pct', 'net_yield_pct', 'price_count', 'rent_count']
    avail = [c for c in display_cols if c in yield_data.columns]
    print(yield_data[avail].to_string(index=False))
else:
    print("Insufficient overlapping data for yield calculation")

# %%
if len(yield_data) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Yield by sub-location
    for beds in sorted(yield_data['beds_numeric'].dropna().unique()):
        sub = yield_data[yield_data['beds_numeric'] == beds]
        axes[0].bar([f"{r['sub_loc_2']}\n{int(beds)}BR" for _, r in sub.iterrows()],
                    sub['gross_yield_pct'], alpha=0.7, label=f'{int(beds)}BR Gross', edgecolor='white')

    axes[0].set_ylabel('Yield %')
    axes[0].set_title('Gross Yield by Segment', fontweight='bold')
    axes[0].legend()
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Gross vs Net comparison
    segments = [f"{r['sub_loc_2']}\n{int(r['beds_numeric'])}BR" for _, r in yield_data.iterrows()]
    x = np.arange(len(segments))
    w = 0.35
    axes[1].bar(x - w/2, yield_data['gross_yield_pct'], w, label='Gross Yield', color='#2BA84A', edgecolor='white')
    axes[1].bar(x + w/2, yield_data['net_yield_pct'], w, label='Net Yield (after SC)', color='#E63946', edgecolor='white')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(segments, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('Yield %')
    axes[1].set_title('Gross vs Net Yield (Service Charge Impact)', fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('../notebooks/fig_03_yield_map.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ### Yield by Location Type (Park-Facing vs Not)

# %%
# Calculate yield for park-facing vs non-park-facing
for label, pf_val in [('Park Facing', 1), ('Not Park Facing', 0)]:
    st = sale_transactions[sale_transactions['is_park_facing'] == pf_val]
    rc = rent_contracts[rent_contracts['is_park_facing'] == pf_val]
    if len(st) > 5 and len(rc) > 5:
        med_price = st['total_sales_price_val'].median()
        med_rent = rc['annualised_rental_price'].median()
        med_size = st['unit_size_sqft'].median()
        sc = med_size * SERVICE_CHARGE_PER_SQFT
        gross = (med_rent / med_price) * 100
        net = ((med_rent - sc) / med_price) * 100
        print(f"\n{label}:")
        print(f"  Median price: AED {med_price:,.0f} | Median rent: AED {med_rent:,.0f}")
        print(f"  Service charge: AED {sc:,.0f}")
        print(f"  Gross yield: {gross:.2f}% | Net yield: {net:.2f}%")
    else:
        print(f"\n{label}: Insufficient data (sales={len(st)}, rentals={len(rc)})")

# %% [markdown]
# ---
# ## Part C: Long-Term Investor Outlook
#
# We model three scenarios — **Bull**, **Base**, and **Bear** — projecting
# total investment returns over 1, 3, and 5 years including:
# - Net rental income (after service charges)
# - Capital appreciation
# - Acquisition costs (DLD 4% + Agent 2%)

# %%
# Monte Carlo Yield Sensitivity
np.random.seed(42)
n_sims = 5000

if len(yield_data) > 0:
    base_price = yield_data['median_price'].mean()
    base_rent = yield_data['median_rent'].mean()
    base_size = yield_data['median_unit_size'].mean()
    base_sc = base_size * SERVICE_CHARGE_PER_SQFT

    rent_changes = np.random.normal(0.05, 0.08, n_sims)  # 5% mean rent growth, 8% std
    price_changes = np.random.normal(0.08, 0.10, n_sims)  # 8% mean price growth, 10% std

    sim_rents = base_rent * (1 + rent_changes)
    sim_prices = base_price * (1 + price_changes)
    sim_gross = (sim_rents / sim_prices) * 100
    sim_net = ((sim_rents - base_sc) / sim_prices) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(sim_net, bins=50, color='#2E86AB', edgecolor='white', alpha=0.7)
    axes[0].axvline(np.median(sim_net), color='#E63946', linewidth=2, linestyle='--',
                     label=f'Median: {np.median(sim_net):.2f}%')
    axes[0].axvline(np.percentile(sim_net, 10), color='#F4A261', linewidth=1.5, linestyle=':',
                     label=f'10th pctile: {np.percentile(sim_net, 10):.2f}%')
    axes[0].axvline(np.percentile(sim_net, 90), color='#2BA84A', linewidth=1.5, linestyle=':',
                     label=f'90th pctile: {np.percentile(sim_net, 90):.2f}%')
    axes[0].set_title('Monte Carlo — Net Yield Distribution (5,000 sims)', fontweight='bold')
    axes[0].set_xlabel('Net Yield %')
    axes[0].legend(fontsize=9)

    # Scenario Analysis
    scenarios = {
        'Bear': {'rent_growth': 0.00, 'price_growth': 0.03},
        'Base': {'rent_growth': 0.05, 'price_growth': 0.08},
        'Bull': {'rent_growth': 0.10, 'price_growth': 0.15},
    }

    years = [1, 2, 3, 4, 5]
    for name, params in scenarios.items():
        cumulative_returns = []
        for yr in years:
            annual_rent = base_rent * (1 + params['rent_growth']) ** yr
            annual_sc = base_sc
            net_income = (annual_rent - annual_sc) * yr
            cap_gain = base_price * ((1 + params['price_growth']) ** yr - 1)
            acq_cost = base_price * (DLD_FEE_PCT + AGENT_COMMISSION_PCT)
            total_return = ((net_income + cap_gain - acq_cost) / (base_price + acq_cost)) * 100
            cumulative_returns.append(total_return)
        axes[1].plot(years, cumulative_returns, 'o-', linewidth=2, markersize=6, label=name)

    axes[1].set_xlabel('Years')
    axes[1].set_ylabel('Cumulative Total Return %')
    axes[1].set_title('Investment Scenarios — Total Return Projection', fontweight='bold')
    axes[1].legend()
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../notebooks/fig_03_scenarios.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print scenario table
    print("\n" + "=" * 70)
    print("5-YEAR INVESTMENT SCENARIO ANALYSIS")
    print("=" * 70)
    print(f"Base assumptions: Purchase AED {base_price:,.0f} | Rent AED {base_rent:,.0f}/yr | SC AED {base_sc:,.0f}/yr")
    print(f"Acquisition costs: DLD {DLD_FEE_PCT*100}% + Agent {AGENT_COMMISSION_PCT*100}% = AED {base_price*(DLD_FEE_PCT+AGENT_COMMISSION_PCT):,.0f}")
    print()
    for name, params in scenarios.items():
        rg, pg = params['rent_growth'], params['price_growth']
        yr5_rent = base_rent * (1+rg)**5
        yr5_income = sum(base_rent*(1+rg)**i - base_sc for i in range(1,6))
        yr5_cap = base_price * ((1+pg)**5 - 1)
        acq = base_price * (DLD_FEE_PCT + AGENT_COMMISSION_PCT)
        total = yr5_income + yr5_cap - acq
        roi = total / (base_price + acq) * 100
        print(f"{name:5s} (rent +{rg*100:.0f}%, price +{pg*100:.0f}%/yr):")
        print(f"  5yr net rental income: AED {yr5_income:>12,.0f}")
        print(f"  5yr capital gain:      AED {yr5_cap:>12,.0f}")
        print(f"  Less acquisition cost: AED {acq:>12,.0f}")
        print(f"  Total 5yr return:      AED {total:>12,.0f} ({roi:.1f}%)")
        print()
else:
    print("⚠️ Insufficient data for scenario analysis")

# %% [markdown]
# ---
# ## Key Takeaways for Investors
#
# 1. **Rent predictability**: Our models predict rents with quantifiable accuracy —
#    use these to validate asking rents on listings.
#
# 2. **Net yield reality**: After the 3.1 AED/sqft service charge, gross yields drop
#    measurably. Always evaluate on net yield, not gross.
#
# 3. **Location premium on rent**: Park-facing properties command a quantifiable rent premium.
#    Compare this premium against the price premium to determine if the extra cost is worth it.
#
# 4. **Long-term scenarios**: Even in the Bear case, the investment breaks even within
#    a reasonable timeframe. The Bull case shows significant upside potential.
