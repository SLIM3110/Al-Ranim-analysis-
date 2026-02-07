"""Feature engineering for Al-Ranim real estate analysis."""

import re
import pandas as pd
import numpy as np
from src.config import SERVICE_CHARGE_PER_SQFT


def extract_comment_features(df, comment_col="comments"):
    """Parse free-text comments into binary location and marketing features."""
    df = df.copy()
    comments = df[comment_col].fillna("").str.lower()

    # Location features
    df["is_park_facing"] = comments.str.contains(
        r"park\s*fac|facing\s*park|park\s*view|overlook.*park", regex=True
    ).astype(int)

    df["is_pool_adjacent"] = comments.str.contains(
        r"pool|swimming", regex=True
    ).astype(int)

    df["is_corner_unit"] = comments.str.contains(
        r"corner", regex=True
    ).astype(int)

    df["is_single_row"] = comments.str.contains(
        r"single\s*row", regex=True
    ).astype(int)

    df["is_end_unit"] = comments.str.contains(
        r"end\s*unit|end\s*row", regex=True
    ).astype(int)

    df["is_near_amenities"] = comments.str.contains(
        r"near\s*amenit|close\s*to\s*amenit|walkway|community\s*cent", regex=True
    ).astype(int)

    df["is_road_facing"] = comments.str.contains(
        r"road\s*fac|facing\s*road|main\s*road", regex=True
    ).astype(int)

    # Marketing features
    df["is_brand_new"] = comments.str.contains(
        r"brand\s*new|never\s*lived|vacant\s*on\s*transfer|ready\s*to\s*move", regex=True
    ).astype(int)

    df["is_motivated_seller"] = comments.str.contains(
        r"motivated\s*seller|urgent|must\s*sell|quick\s*sale", regex=True
    ).astype(int)

    df["is_hot_deal"] = comments.str.contains(
        r"hot\s*deal|great\s*deal|best\s*price|below\s*market|reduced", regex=True
    ).astype(int)

    df["has_big_plot"] = comments.str.contains(
        r"big\s*plot|large\s*plot|huge\s*plot|oversized\s*plot", regex=True
    ).astype(int)

    df["has_mortgage_mention"] = comments.str.contains(
        r"mortgage|bank\s*financ|loan", regex=True
    ).astype(int)

    df["is_upgraded"] = comments.str.contains(
        r"upgrad|renovat|modern|landscap", regex=True
    ).astype(int)

    df["is_rented"] = comments.str.contains(
        r"rented|tenant|occupied|leased", regex=True
    ).astype(int)

    return df


def normalize_developer_type(df, col="developer_type"):
    """Standardize messy developer_type into unit_subtype and unit_position."""
    df = df.copy()
    raw = df[col].fillna("").str.strip()

    # Extract unit subtype (e.g., 3B1, 3B2, 4B1, 4B2)
    subtype_pattern = r"(\d[Bb]\d)"
    df["unit_subtype"] = raw.str.extract(subtype_pattern, expand=False)
    df["unit_subtype"] = df["unit_subtype"].str.upper()

    # Extract unit position
    lower = raw.str.lower()
    df["unit_position"] = "unknown"
    df.loc[lower.str.contains("mid", na=False), "unit_position"] = "mid"
    df.loc[lower.str.contains("end", na=False), "unit_position"] = "end"
    df.loc[lower.str.contains("corner", na=False), "unit_position"] = "corner"

    # Check if (M) suffix indicates maid room
    df["has_maid_room"] = raw.str.contains(r"\(M\)", case=False, na=False).astype(int)

    return df


def create_sales_features(df):
    """Engineer features for sales analysis."""
    df = df.copy()

    # Plot to unit ratio
    if "plot_size_sqft" in df.columns and "unit_size_sqft" in df.columns:
        df["plot_to_unit_ratio"] = df["plot_size_sqft"] / df["unit_size_sqft"]
        df["plot_to_unit_ratio"] = df["plot_to_unit_ratio"].replace(
            [np.inf, -np.inf], np.nan
        )

    # Floor level encoding
    if "floor_level" in df.columns:
        df["is_g_plus_2"] = (df["floor_level"] == "G+2").astype(int)

    # Resale flag
    if "sale_sequence" in df.columns:
        df["is_resale"] = (df["sale_sequence"] == "Resale").astype(int)

    # Beds as numeric
    if "no_beds" in df.columns:
        df["beds_numeric"] = pd.to_numeric(df["no_beds"], errors="coerce")

    return df


def create_rental_features(df):
    """Engineer features for rental analysis."""
    df = df.copy()

    # Furnished flag
    if "furnished" in df.columns:
        df["is_furnished"] = df["furnished"].str.strip().str.lower().isin(
            ["yes", "y", "furnished"]
        ).astype(int)

    # Has view
    if "views" in df.columns:
        df["has_view"] = df["views"].notna().astype(int)

    # Floor level encoding
    if "floor_level" in df.columns:
        df["is_g_plus_2"] = (df["floor_level"] == "G+2").astype(int)

    # Beds as numeric
    beds_col = "beds" if "beds" in df.columns else "no_beds"
    if beds_col in df.columns:
        df["beds_numeric"] = pd.to_numeric(df[beds_col], errors="coerce")

    # Rent per sqft
    size_col = "unit_size" if "unit_size" in df.columns else "unit_size_sqft"
    rent_col = "annualised_rental_price"
    if rent_col in df.columns and size_col in df.columns:
        df["rent_per_sqft"] = df[rent_col] / df[size_col]
        df["rent_per_sqft"] = df["rent_per_sqft"].replace([np.inf, -np.inf], np.nan)

    return df


def compute_service_charge(unit_size_sqft):
    """Compute annual service charge for a given unit size."""
    return unit_size_sqft * SERVICE_CHARGE_PER_SQFT


def compute_yield_metrics(sales_df, rental_df, group_cols=None):
    """Calculate rental yield by matching property segments.

    Uses median values from actual transactions/contracts only.
    Returns gross yield, net yield (after service charges), and total cost metrics.
    """
    if group_cols is None:
        group_cols = ["sub_loc_2", "beds_numeric"]

    # Ensure beds_numeric exists
    if "beds_numeric" not in sales_df.columns:
        beds_col = "no_beds" if "no_beds" in sales_df.columns else "beds"
        if beds_col in sales_df.columns:
            sales_df = sales_df.copy()
            sales_df["beds_numeric"] = pd.to_numeric(sales_df[beds_col], errors="coerce")

    if "beds_numeric" not in rental_df.columns:
        beds_col = "beds" if "beds" in rental_df.columns else "no_beds"
        if beds_col in rental_df.columns:
            rental_df = rental_df.copy()
            rental_df["beds_numeric"] = pd.to_numeric(rental_df[beds_col], errors="coerce")

    # Filter valid group columns
    valid_sales_cols = [c for c in group_cols if c in sales_df.columns]
    valid_rental_cols = [c for c in group_cols if c in rental_df.columns]

    if not valid_sales_cols or not valid_rental_cols:
        return pd.DataFrame()

    # Sales aggregation
    price_col = "total_sales_price_val"
    sales_agg = (
        sales_df.groupby(valid_sales_cols)
        .agg(
            median_price=(price_col, "median"),
            mean_price=(price_col, "mean"),
            price_count=(price_col, "count"),
            median_unit_size=("unit_size_sqft", "median"),
        )
        .reset_index()
    )

    # Rental aggregation
    rent_col = "annualised_rental_price"
    size_col = "unit_size" if "unit_size" in rental_df.columns else "unit_size_sqft"
    rental_agg = (
        rental_df.groupby(valid_rental_cols)
        .agg(
            median_rent=(rent_col, "median"),
            mean_rent=(rent_col, "mean"),
            rent_count=(rent_col, "count"),
        )
        .reset_index()
    )

    # Merge
    merged = sales_agg.merge(rental_agg, on=valid_sales_cols, how="inner")

    # Yield calculations
    merged["gross_yield_pct"] = (merged["median_rent"] / merged["median_price"]) * 100
    merged["annual_service_charge"] = merged["median_unit_size"] * SERVICE_CHARGE_PER_SQFT
    merged["net_rent"] = merged["median_rent"] - merged["annual_service_charge"]
    merged["net_yield_pct"] = (merged["net_rent"] / merged["median_price"]) * 100

    # Total acquisition cost
    from src.config import DLD_FEE_PCT, AGENT_COMMISSION_PCT
    merged["total_acquisition_cost"] = merged["median_price"] * (1 + DLD_FEE_PCT + AGENT_COMMISSION_PCT)
    merged["net_yield_on_total_cost_pct"] = (merged["net_rent"] / merged["total_acquisition_cost"]) * 100

    return merged
