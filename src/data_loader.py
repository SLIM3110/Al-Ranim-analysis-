"""Data loading, cleaning, and preparation for Al-Ranim analysis."""

import pandas as pd
import numpy as np
from src.config import (
    SALES_CSV, RENTAL_CSV, CSV_ENCODING, EXCLUDED_UNIT_TYPES,
    ACTUAL_SALES, ACTUAL_RENTALS, LISTINGS_SALES, LISTINGS_RENTALS,
    SERVICE_CHARGE_PER_SQFT,
)


def load_sales_data(path=None):
    """Load sales CSV with latin-1 encoding, parse dates, fix types, exclude apartments."""
    if path is None:
        path = SALES_CSV

    df = pd.read_csv(path, encoding=CSV_ENCODING)

    # Replace string 'NULL' with NaN
    df.replace("NULL", np.nan, inplace=True)
    df.replace("null", np.nan, inplace=True)

    # Exclude apartments
    df = df[~df["unit_type"].isin(EXCLUDED_UNIT_TYPES)].copy()

    # Parse dates
    df["custom_date"] = pd.to_datetime(df["custom_date"], errors="coerce")

    # Convert numeric columns
    numeric_cols = [
        "total_sales_price_val", "sales_price_sqft_unit", "sales_price_sqm_unit",
        "unit_size_sqft", "unit_size_sqm", "plot_size_sqft", "plot_size_sqm",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace 0 in plot_size with NaN
    if "plot_size_sqft" in df.columns:
        df.loc[df["plot_size_sqft"] == 0, "plot_size_sqft"] = np.nan
    if "plot_size_sqm" in df.columns:
        df.loc[df["plot_size_sqm"] == 0, "plot_size_sqm"] = np.nan

    # Add data source classification
    df["data_source"] = "listing"
    df.loc[df["evdnc_name"].isin(ACTUAL_SALES), "data_source"] = "transaction"
    df.loc[df["evdnc_name"].str.contains("Pending", na=False), "data_source"] = "pending"

    # Add time features
    if df["custom_date"].notna().any():
        min_date = df["custom_date"].min()
        df["months_since_start"] = (
            (df["custom_date"] - min_date).dt.days / 30.44
        ).round(1)
        df["month"] = df["custom_date"].dt.to_period("M")
        df["year_month"] = df["custom_date"].dt.strftime("%Y-%m")

    # Compute service charge
    df["annual_service_charge"] = df["unit_size_sqft"] * SERVICE_CHARGE_PER_SQFT

    df = df.reset_index(drop=True)
    return df


def load_rental_data(path=None):
    """Load rental CSV with latin-1 encoding, parse dates, fix types, exclude apartments."""
    if path is None:
        path = RENTAL_CSV

    df = pd.read_csv(path, encoding=CSV_ENCODING)

    # Replace string 'NULL' with NaN
    df.replace("NULL", np.nan, inplace=True)
    df.replace("null", np.nan, inplace=True)

    # Exclude apartments
    if "unit_type" in df.columns:
        df = df[~df["unit_type"].isin(EXCLUDED_UNIT_TYPES)].copy()

    # Parse dates
    for col in ["start_date", "end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Convert numeric columns
    numeric_cols = [
        "annualised_rental_price", "contract_rental_price",
        "rent_price_sqft_unit", "rent_price_sqm_unit",
        "unit_size", "unit_size_sqm", "plot_size", "plot_size_sqm",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add data source classification
    df["data_source"] = "listing"
    df.loc[df["evdnc_name"].isin(ACTUAL_RENTALS), "data_source"] = "contract"

    # Add time features
    date_col = "start_date"
    if date_col in df.columns and df[date_col].notna().any():
        min_date = df[date_col].min()
        df["months_since_start"] = (
            (df[date_col] - min_date).dt.days / 30.44
        ).round(1)
        df["month"] = df[date_col].dt.to_period("M")
        df["year_month"] = df[date_col].dt.strftime("%Y-%m")

    # Compute service charge
    size_col = "unit_size" if "unit_size" in df.columns else "unit_size_sqft"
    if size_col in df.columns:
        df["annual_service_charge"] = df[size_col] * SERVICE_CHARGE_PER_SQFT

    # Contract duration
    if "start_date" in df.columns and "end_date" in df.columns:
        df["contract_duration_days"] = (df["end_date"] - df["start_date"]).dt.days
        df["contract_duration_months"] = (df["contract_duration_days"] / 30.44).round(1)

    df = df.reset_index(drop=True)
    return df


def get_transactions_only(df):
    """Filter to actual closed transactions."""
    return df[df["data_source"] == "transaction"].copy()


def get_contracts_only(df):
    """Filter to actual rental contracts."""
    return df[df["data_source"] == "contract"].copy()


def get_listings_only(df):
    """Filter to active listings."""
    return df[df["data_source"] == "listing"].copy()
