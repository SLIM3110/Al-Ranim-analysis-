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
# # 🏷️ Sale Price Prediction — 4 ML Models
# ## Al-Ranim, Mudon, Dubai
#
# This notebook builds **4 machine learning models** that each approach price prediction differently.
# We train exclusively on **actual transactions** (Oqood + Title Deed) — never on listing prices.
#
# | Model | How It Thinks | What Investors Learn |
# |-------|--------------|---------------------|
# | **Ridge Regression** | Linear, additive effects | Exact AED premium for each feature |
# | **Random Forest** | Non-linear interactions | Which feature combos matter most |
# | **XGBoost** | Sequential error correction | Best accuracy + confidence intervals |
# | **KNN** | Find similar past sales | "What did comparable properties sell for?" |

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
import time

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from src.config import *
from src.data_loader import load_sales_data, get_transactions_only, get_listings_only
from src.feature_engineering import extract_comment_features, normalize_developer_type, create_sales_features
from src.visualization import setup_style, format_aed

setup_style()
print("✅ All libraries loaded")

# %% [markdown]
# ## 1. Data Preparation

# %%
# Load and engineer features
sales = load_sales_data()
sales = extract_comment_features(sales)
sales = normalize_developer_type(sales)
sales = create_sales_features(sales)

# Split into transactions (training) and listings (prediction)
transactions = get_transactions_only(sales)
listings = get_listings_only(sales)

print(f"📊 Training data (actual transactions): {len(transactions):,} rows")
print(f"📊 Prediction data (active listings):   {len(listings):,} rows")

# %%
# Define features
numeric_features = ['unit_size_sqft', 'plot_size_sqft', 'months_since_start']
categorical_features = ['sub_loc_2']
binary_features = ['beds_numeric', 'is_g_plus_2', 'is_park_facing', 'is_corner_unit',
                   'is_single_row', 'is_end_unit', 'is_pool_adjacent', 'is_road_facing',
                   'is_brand_new', 'is_upgraded', 'is_resale']

all_features = numeric_features + categorical_features + binary_features
target = 'total_sales_price_val'

# Prepare training data
df_train = transactions[all_features + [target]].copy()

# Impute plot_size_sqft with median per beds group
for beds in df_train['beds_numeric'].unique():
    mask = (df_train['beds_numeric'] == beds) & (df_train['plot_size_sqft'].isna())
    median_val = df_train.loc[df_train['beds_numeric'] == beds, 'plot_size_sqft'].median()
    if pd.isna(median_val):
        median_val = df_train['plot_size_sqft'].median()
    df_train.loc[mask, 'plot_size_sqft'] = median_val

# Drop remaining NaN rows
df_train = df_train.dropna(subset=[target] + numeric_features)

X = df_train[all_features]
y = df_train[target]

print(f"\n✅ Final training set: {X.shape[0]} rows × {X.shape[1]} features")
print(f"   Target range: AED {y.min():,.0f} — AED {y.max():,.0f}")
print(f"   Target median: AED {y.median():,.0f}")

# %%
# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='infrequent_if_exist'), categorical_features),
        ('bin', 'passthrough', binary_features),
    ]
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

# %% [markdown]
# ---
# ## 2. Model 1: Ridge Regression
# *"Each feature adds a fixed AED amount to the price."*
#
# Ridge regression gives us the **most interpretable** model — we can read off exactly
# how many AED each feature adds or subtracts from the price.

# %%
t0 = time.time()
ridge_pipe = Pipeline([('pre', preprocessor), ('model', Ridge())])

# Tune alpha
param_grid = {'model__alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}
ridge_cv = GridSearchCV(ridge_pipe, param_grid, cv=kf, scoring='r2', n_jobs=-1)
ridge_cv.fit(X, y)
best_ridge = ridge_cv.best_estimator_
ridge_time = time.time() - t0

# Cross-validation scores
ridge_r2 = cross_val_score(best_ridge, X, y, cv=kf, scoring='r2')
ridge_rmse = np.sqrt(-cross_val_score(best_ridge, X, y, cv=kf, scoring='neg_mean_squared_error'))
ridge_mae = -cross_val_score(best_ridge, X, y, cv=kf, scoring='neg_mean_absolute_error')

results['Ridge'] = {
    'R2': ridge_r2.mean(), 'RMSE': ridge_rmse.mean(), 'MAE': ridge_mae.mean(), 'Time': ridge_time
}

print(f"Ridge Regression (alpha={ridge_cv.best_params_['model__alpha']})")
print(f"  R²:   {ridge_r2.mean():.4f} (±{ridge_r2.std():.4f})")
print(f"  RMSE: AED {ridge_rmse.mean():,.0f}")
print(f"  MAE:  AED {ridge_mae.mean():,.0f}")

# %%
# Extract coefficients
best_ridge.fit(X, y)
feature_names_out = (best_ridge.named_steps['pre']
                     .get_feature_names_out()
                     if hasattr(best_ridge.named_steps['pre'], 'get_feature_names_out')
                     else all_features)
# Clean up names
feature_names_clean = [str(f).replace('num__', '').replace('cat__', '').replace('bin__', '')
                       for f in feature_names_out]
coefficients = best_ridge.named_steps['model'].coef_

# Sort by absolute value
coef_df = pd.DataFrame({'Feature': feature_names_clean, 'Coefficient': coefficients})
coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values('Abs_Coef', ascending=True)

fig, ax = plt.subplots(figsize=(12, max(6, len(coef_df) * 0.35)))
colors = ['#2BA84A' if c > 0 else '#E63946' for c in coef_df['Coefficient']]
ax.barh(range(len(coef_df)), coef_df['Coefficient'], color=colors, edgecolor='white')
ax.set_yticks(range(len(coef_df)))
ax.set_yticklabels(coef_df['Feature'])
ax.set_xlabel('Coefficient (AED impact on price)')
ax.set_title('Ridge Regression — Feature Coefficients\n(Green = increases price, Red = decreases price)',
             fontweight='bold')
ax.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('../notebooks/fig_02_ridge_coefficients.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Ridge Regression — Key Investor Takeaways

# %%
# Print top positive and negative coefficients
print("=" * 60)
print("RIDGE REGRESSION — PRICE DRIVERS")
print("=" * 60)
sorted_coefs = coef_df.sort_values('Coefficient', ascending=False)
print("\n🔼 TOP PRICE INCREASERS:")
for _, row in sorted_coefs.head(5).iterrows():
    print(f"   {row['Feature']:30s} → +AED {row['Coefficient']:>12,.0f}")
print("\n🔽 TOP PRICE DECREASERS:")
for _, row in sorted_coefs.tail(5).iterrows():
    print(f"   {row['Feature']:30s} → AED {row['Coefficient']:>12,.0f}")

# %%
# Actual vs Predicted
y_pred_ridge = best_ridge.predict(X)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y, y_pred_ridge, alpha=0.4, s=30, color='#2E86AB')
min_val, max_val = min(y.min(), y_pred_ridge.min()), max(y.max(), y_pred_ridge.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
axes[0].set_xlabel('Actual Price (AED)')
axes[0].set_ylabel('Predicted Price (AED)')
axes[0].set_title('Ridge: Actual vs Predicted', fontweight='bold')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
axes[0].legend()

# Residuals
residuals = y - y_pred_ridge
axes[1].scatter(y_pred_ridge, residuals, alpha=0.4, s=30, color='#E63946')
axes[1].axhline(0, color='black', linewidth=1)
axes[1].set_xlabel('Predicted Price (AED)')
axes[1].set_ylabel('Residual (AED)')
axes[1].set_title('Ridge: Residuals', fontweight='bold')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(format_aed))

plt.tight_layout()
plt.savefig('../notebooks/fig_02_ridge_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 3. Model 2: Random Forest
# *"Feature interactions matter — the value of park-facing may depend on the number of bedrooms."*

# %%
t0 = time.time()
rf_pipe = Pipeline([('pre', preprocessor), ('model', RandomForestRegressor(random_state=42, n_jobs=-1))])

param_grid_rf = {
    'model__n_estimators': [100, 300],
    'model__max_depth': [8, 12, None],
    'model__min_samples_leaf': [3, 5],
}
rf_cv = GridSearchCV(rf_pipe, param_grid_rf, cv=kf, scoring='r2', n_jobs=-1)
rf_cv.fit(X, y)
best_rf = rf_cv.best_estimator_
rf_time = time.time() - t0

rf_r2 = cross_val_score(best_rf, X, y, cv=kf, scoring='r2')
rf_rmse = np.sqrt(-cross_val_score(best_rf, X, y, cv=kf, scoring='neg_mean_squared_error'))
rf_mae = -cross_val_score(best_rf, X, y, cv=kf, scoring='neg_mean_absolute_error')

results['Random Forest'] = {
    'R2': rf_r2.mean(), 'RMSE': rf_rmse.mean(), 'MAE': rf_mae.mean(), 'Time': rf_time
}

print(f"Random Forest (best params: {rf_cv.best_params_})")
print(f"  R²:   {rf_r2.mean():.4f} (±{rf_r2.std():.4f})")
print(f"  RMSE: AED {rf_rmse.mean():,.0f}")
print(f"  MAE:  AED {rf_mae.mean():,.0f}")

# %%
# Feature importance (MDI)
best_rf.fit(X, y)
importances = best_rf.named_steps['model'].feature_importances_
feat_names = feature_names_clean if len(feature_names_clean) == len(importances) else [f'f{i}' for i in range(len(importances))]

fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
fi_df = fi_df.sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(12, max(6, len(fi_df) * 0.35)))
ax.barh(range(len(fi_df)), fi_df['Importance'], color='#2E86AB', edgecolor='white')
ax.set_yticks(range(len(fi_df)))
ax.set_yticklabels(fi_df['Feature'])
ax.set_xlabel('Feature Importance (MDI)')
ax.set_title('Random Forest — Feature Importance', fontweight='bold')
plt.tight_layout()
plt.savefig('../notebooks/fig_02_rf_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Actual vs Predicted
y_pred_rf = best_rf.predict(X)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y, y_pred_rf, alpha=0.4, s=30, color='#2BA84A')
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
ax.set_xlabel('Actual Price (AED)')
ax.set_ylabel('Predicted Price (AED)')
ax.set_title('Random Forest: Actual vs Predicted', fontweight='bold')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
plt.tight_layout()
plt.savefig('../notebooks/fig_02_rf_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 4. Model 3: XGBoost & LightGBM
# *"Maximum accuracy through sequential error correction."*

# %%
t0 = time.time()
X_processed = preprocessor.fit_transform(X)

# XGBoost
xgb_model = xgb.XGBRegressor(
    learning_rate=0.05, n_estimators=500, max_depth=4, subsample=0.8,
    colsample_bytree=0.8, random_state=42, n_jobs=-1
)
xgb_r2 = cross_val_score(xgb_model, X_processed, y, cv=kf, scoring='r2')
xgb_rmse = np.sqrt(-cross_val_score(xgb_model, X_processed, y, cv=kf, scoring='neg_mean_squared_error'))
xgb_mae = -cross_val_score(xgb_model, X_processed, y, cv=kf, scoring='neg_mean_absolute_error')
xgb_time = time.time() - t0

results['XGBoost'] = {
    'R2': xgb_r2.mean(), 'RMSE': xgb_rmse.mean(), 'MAE': xgb_mae.mean(), 'Time': xgb_time
}

print(f"XGBoost:")
print(f"  R²:   {xgb_r2.mean():.4f} (±{xgb_r2.std():.4f})")
print(f"  RMSE: AED {xgb_rmse.mean():,.0f}")
print(f"  MAE:  AED {xgb_mae.mean():,.0f}")

# LightGBM
t0 = time.time()
lgb_model = lgb.LGBMRegressor(
    learning_rate=0.05, n_estimators=500, max_depth=4, subsample=0.8,
    colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
)
lgb_r2 = cross_val_score(lgb_model, X_processed, y, cv=kf, scoring='r2')
lgb_rmse = np.sqrt(-cross_val_score(lgb_model, X_processed, y, cv=kf, scoring='neg_mean_squared_error'))
lgb_mae = -cross_val_score(lgb_model, X_processed, y, cv=kf, scoring='neg_mean_absolute_error')
lgb_time = time.time() - t0

results['LightGBM'] = {
    'R2': lgb_r2.mean(), 'RMSE': lgb_rmse.mean(), 'MAE': lgb_mae.mean(), 'Time': lgb_time
}

print(f"\nLightGBM:")
print(f"  R²:   {lgb_r2.mean():.4f} (±{lgb_r2.std():.4f})")
print(f"  RMSE: AED {lgb_rmse.mean():,.0f}")
print(f"  MAE:  AED {lgb_mae.mean():,.0f}")

# %%
# Fit best boosting model and get SHAP values
best_boost = xgb_model if xgb_r2.mean() >= lgb_r2.mean() else lgb_model
best_boost_name = 'XGBoost' if xgb_r2.mean() >= lgb_r2.mean() else 'LightGBM'
best_boost.fit(X_processed, y)

try:
    import shap
    explainer = shap.TreeExplainer(best_boost)
    shap_values = explainer.shap_values(X_processed)

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_processed, feature_names=feat_names,
                      show=False, max_display=15)
    plt.title(f'{best_boost_name} — SHAP Feature Importance', fontweight='bold')
    plt.tight_layout()
    plt.savefig('../notebooks/fig_02_shap_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"SHAP visualization skipped: {e}")

# %%
# XGBoost prediction intervals using quantile regression
try:
    xgb_low = xgb.XGBRegressor(
        objective='reg:quantileerror', quantile_alpha=0.1,
        learning_rate=0.05, n_estimators=300, max_depth=4, random_state=42
    )
    xgb_high = xgb.XGBRegressor(
        objective='reg:quantileerror', quantile_alpha=0.9,
        learning_rate=0.05, n_estimators=300, max_depth=4, random_state=42
    )
    xgb_low.fit(X_processed, y)
    xgb_high.fit(X_processed, y)

    y_pred_med = best_boost.predict(X_processed)
    y_pred_low = xgb_low.predict(X_processed)
    y_pred_high = xgb_high.predict(X_processed)

    # Sort for plotting
    sort_idx = np.argsort(y.values)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(range(len(y)), y_pred_low[sort_idx], y_pred_high[sort_idx],
                     alpha=0.3, color='#2E86AB', label='80% Prediction Interval')
    ax.plot(range(len(y)), y_pred_med[sort_idx], color='#2E86AB', linewidth=1.5, label='Median Prediction')
    ax.scatter(range(len(y)), y.values[sort_idx], s=15, color='#E63946', alpha=0.5, label='Actual')
    ax.set_xlabel('Properties (sorted by actual price)')
    ax.set_ylabel('Price (AED)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
    ax.set_title(f'{best_boost_name} — Prediction Intervals (80%)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('../notebooks/fig_02_prediction_intervals.png', dpi=150, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Quantile regression skipped: {e}")

# %% [markdown]
# ---
# ## 5. Model 4: KNN (Comparable Sales)
# *"What did the 5 most similar properties actually sell for?"*
#
# This is the model closest to how real estate agents work — finding comparable sales ("comps").

# %%
# Optimize k
t0 = time.time()
k_range = range(3, 16)
k_scores = []
for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_processed, y, cv=kf, scoring='r2')
    k_scores.append(scores.mean())

best_k = list(k_range)[np.argmax(k_scores)]
knn_best = KNeighborsRegressor(n_neighbors=best_k)
knn_r2 = cross_val_score(knn_best, X_processed, y, cv=kf, scoring='r2')
knn_rmse = np.sqrt(-cross_val_score(knn_best, X_processed, y, cv=kf, scoring='neg_mean_squared_error'))
knn_mae = -cross_val_score(knn_best, X_processed, y, cv=kf, scoring='neg_mean_absolute_error')
knn_time = time.time() - t0

results['KNN'] = {
    'R2': knn_r2.mean(), 'RMSE': knn_rmse.mean(), 'MAE': knn_mae.mean(), 'Time': knn_time
}

print(f"KNN (k={best_k}):")
print(f"  R²:   {knn_r2.mean():.4f} (±{knn_r2.std():.4f})")
print(f"  RMSE: AED {knn_rmse.mean():,.0f}")
print(f"  MAE:  AED {knn_mae.mean():,.0f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(list(k_range), k_scores, 'o-', color='#2E86AB', linewidth=2, markersize=8)
ax.axvline(best_k, color='#E63946', linestyle='--', label=f'Best k={best_k}')
ax.set_xlabel('k (Number of Neighbors)')
ax.set_ylabel('R² Score')
ax.set_title('KNN — Optimal k Selection', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('../notebooks/fig_02_knn_k_selection.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Comparable sales demo
knn_best.fit(X_processed, y)

def find_comparable_sales(property_idx, k=5):
    """Find k most comparable actual transactions."""
    distances, indices = knn_best.kneighbors(X_processed[property_idx:property_idx+1], n_neighbors=k)
    comps = df_train.iloc[indices[0]].copy()
    comps['distance'] = distances[0]
    return comps

# Demo for a sample property
print("=" * 60)
print(f"COMPARABLE SALES DEMO (k={best_k})")
print("=" * 60)

sample_idx = 0
sample = df_train.iloc[sample_idx]
print(f"\nTarget property:")
print(f"  Location: {sample.get('sub_loc_2', 'N/A')}")
print(f"  Beds: {sample.get('beds_numeric', 'N/A')}")
print(f"  Size: {sample.get('unit_size_sqft', 'N/A'):,.0f} sqft")
print(f"  Price: AED {sample[target]:,.0f}")

comps = find_comparable_sales(sample_idx, best_k)
print(f"\n{best_k} Most Comparable Sales:")
for i, (_, comp) in enumerate(comps.iterrows()):
    print(f"  {i+1}. AED {comp[target]:,.0f} | {comp.get('sub_loc_2', 'N/A')} | "
          f"{comp.get('beds_numeric', 'N/A')}BR | {comp.get('unit_size_sqft', 0):,.0f} sqft")

# %% [markdown]
# ---
# ## 6. Model Comparison

# %%
print("=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
results_df = pd.DataFrame(results).T
results_df['RMSE'] = results_df['RMSE'].round(0)
results_df['MAE'] = results_df['MAE'].round(0)
results_df['R2'] = results_df['R2'].round(4)
results_df['Time'] = results_df['Time'].round(2)
print(results_df.to_string())

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
models = list(results.keys())
colors = sns.color_palette('husl', len(models))

for ax, metric, title in zip(axes, ['R2', 'RMSE', 'MAE'], ['R² (Higher is Better)', 'RMSE (Lower is Better)', 'MAE (Lower is Better)']):
    vals = [results[m][metric] for m in models]
    bars = ax.bar(range(len(models)), vals, color=colors, edgecolor='white')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.set_title(title, fontweight='bold')
    if metric in ['RMSE', 'MAE']:
        ax.bar_label(bars, [f'AED {v:,.0f}' for v in vals], padding=3, fontsize=8)
    else:
        ax.bar_label(bars, [f'{v:.3f}' for v in vals], padding=3, fontsize=9)

plt.suptitle('Model Performance Comparison', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../notebooks/fig_02_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 7. Location Premium Analysis (Deep Dive)
#
# **The key question: How much more do park-facing, corner, and single-row properties cost?**

# %%
# Location premium from Ridge coefficients
location_feats = ['is_park_facing', 'is_corner_unit', 'is_single_row', 'is_end_unit',
                  'is_pool_adjacent', 'is_road_facing']

print("=" * 60)
print("LOCATION PREMIUM ANALYSIS")
print("=" * 60)

premium_data = []
for feat in location_feats:
    if feat in coef_df['Feature'].values:
        coef_val = coef_df.loc[coef_df['Feature'] == feat, 'Coefficient'].values[0]
        median_price = y.median()
        pct_premium = (coef_val / median_price) * 100
        premium_data.append({
            'Feature': feat.replace('is_', '').replace('_', ' ').title(),
            'AED Premium': coef_val,
            '% Premium': pct_premium,
        })
        print(f"  {feat:25s} → AED {coef_val:>12,.0f} ({pct_premium:+.1f}%)")

premium_df = pd.DataFrame(premium_data)

fig, ax = plt.subplots(figsize=(10, 5))
colors_loc = ['#2BA84A' if p > 0 else '#E63946' for p in premium_df['AED Premium']]
bars = ax.barh(premium_df['Feature'], premium_df['AED Premium'], color=colors_loc, edgecolor='white')
ax.set_xlabel('Price Premium (AED)')
ax.set_title('Location Premium — Ridge Regression Coefficients', fontweight='bold')
ax.axvline(0, color='black', linewidth=0.5)
ax.bar_label(bars, [f'AED {v:,.0f}' for v in premium_df['AED Premium']], padding=5, fontsize=9)
plt.tight_layout()
plt.savefig('../notebooks/fig_02_location_premium.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Park-facing impact by beds
fig, ax = plt.subplots(figsize=(10, 6))
for beds in sorted(df_train['beds_numeric'].dropna().unique()):
    for pf, label, color in [(0, 'Not Park Facing', '#F4A261'), (1, 'Park Facing', '#2BA84A')]:
        subset = df_train[(df_train['beds_numeric'] == beds) & (df_train['is_park_facing'] == pf)]
        if len(subset) > 2:
            ax.bar(f'{int(beds)}BR\n{label}', subset[target].median(),
                   color=color, edgecolor='white', width=0.6)

ax.set_ylabel('Median Sale Price (AED)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
ax.set_title('Median Sale Price: Park-Facing vs Not, by Bedrooms', fontweight='bold')
plt.tight_layout()
plt.savefig('../notebooks/fig_02_park_by_beds.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 8. Listing Fair-Value Scoring
#
# Using our best model, we predict "fair value" for every active listing and flag those
# that appear **overpriced** or **undervalued** relative to actual transaction data.

# %%
# Prepare listings for prediction
listings_prep = listings[all_features].copy()
for beds in listings_prep['beds_numeric'].unique():
    mask = (listings_prep['beds_numeric'] == beds) & (listings_prep['plot_size_sqft'].isna())
    median_val = listings_prep.loc[listings_prep['beds_numeric'] == beds, 'plot_size_sqft'].median()
    if pd.isna(median_val):
        median_val = df_train['plot_size_sqft'].median()
    listings_prep.loc[mask, 'plot_size_sqft'] = median_val

listings_prep = listings_prep.dropna(subset=numeric_features)
listings_processed = preprocessor.transform(listings_prep)

# Predict with best model
best_boost.fit(X_processed, y)
listings_predicted = best_boost.predict(listings_processed)
listings_actual = listings.loc[listings_prep.index, target].values

# Calculate overpricing percentage
overpriced_pct = ((listings_actual - listings_predicted) / listings_predicted) * 100

# Categorize
categories = pd.cut(overpriced_pct, bins=[-np.inf, -5, 10, 15, np.inf],
                     labels=['Undervalued (<-5%)', 'Fair (±10%)', 'Slightly Over (10-15%)', 'Overpriced (>15%)'])

print("=" * 60)
print("LISTING FAIR-VALUE SCORING")
print("=" * 60)
print(f"\nTotal listings scored: {len(overpriced_pct):,}")
print(f"\nCategory breakdown:")
print(categories.value_counts().to_string())

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution of overpricing
axes[0].hist(overpriced_pct, bins=40, color='#2E86AB', edgecolor='white', alpha=0.7)
axes[0].axvline(0, color='#E63946', linewidth=2, linestyle='--', label='Fair Value')
axes[0].set_xlabel('Overpricing %')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Listing Overpricing %', fontweight='bold')
axes[0].legend()

# Category pie chart
cat_counts = categories.value_counts()
colors_pie = ['#2BA84A', '#2E86AB', '#F4A261', '#E63946']
axes[1].pie(cat_counts.values, labels=cat_counts.index, colors=colors_pie,
            autopct='%1.1f%%', startangle=90)
axes[1].set_title('Listing Categories', fontweight='bold')

plt.tight_layout()
plt.savefig('../notebooks/fig_02_listing_scoring.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Top undervalued and overpriced listings
scoring_df = listings.loc[listings_prep.index].copy()
scoring_df['predicted_price'] = listings_predicted
scoring_df['overpriced_pct'] = overpriced_pct
scoring_df['category'] = list(categories)

print("\n🟢 TOP 10 POTENTIALLY UNDERVALUED LISTINGS:")
print("   (Listed below model-predicted fair value)")
undervalued = scoring_df.nsmallest(10, 'overpriced_pct')[
    ['sub_loc_2', 'no_beds', 'unit_size_sqft', 'total_sales_price_val', 'predicted_price', 'overpriced_pct']
].copy()
undervalued.columns = ['Location', 'Beds', 'Size(sqft)', 'Listing Price', 'Fair Value', 'Over/Under %']
print(undervalued.to_string(index=False))

print("\n🔴 TOP 10 MOST OVERPRICED LISTINGS:")
overpriced = scoring_df.nlargest(10, 'overpriced_pct')[
    ['sub_loc_2', 'no_beds', 'unit_size_sqft', 'total_sales_price_val', 'predicted_price', 'overpriced_pct']
].copy()
overpriced.columns = ['Location', 'Beds', 'Size(sqft)', 'Listing Price', 'Fair Value', 'Over/Under %']
print(overpriced.to_string(index=False))

# %% [markdown]
# ---
# ## Key Takeaways for Investors
#
# 1. **Model accuracy**: Our best model predicts actual transaction prices with measurable precision.
#    Use the R² and MAE metrics above to gauge confidence.
#
# 2. **Location premiums**: Park-facing and corner properties command quantifiable premiums.
#    The Ridge model gives exact AED amounts; SHAP shows how these interact with other features.
#
# 3. **Listing overpricing**: A significant portion of active listings are priced above fair value
#    based on actual transaction data. Use the fair-value scoring to negotiate.
#
# 4. **Comparable sales (KNN)**: For any specific property you're considering, the KNN model
#    can find the most similar past transactions — this is your best negotiation tool.
