"""Central configuration for Al-Ranim Real Estate Investment Analysis."""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CSV file paths
SALES_CSV = os.path.join(
    BASE_DIR,
    "al-ranim-market-data-mhaegan-aclan-eva-real-estate-llc-06-02-2026-3f60f8993287553695dd0fb684d5abb8d74fea88.csv",
)
RENTAL_CSV = os.path.join(
    BASE_DIR,
    "al-ranim-market-data-mhaegan-aclan-eva-real-estate-llc-07-02-2026-8db2e10df28520d235f7f1a9669b02e00ffe0429.csv",
)

# Evidence type categories
ACTUAL_SALES = ["Oqood", "Title Deed"]
PENDING_SALES = ["Pending Sales (SPA/MOU)"]
LISTINGS_SALES = ["Active Listings"]
ACTUAL_RENTALS = ["Rental Contracts"]
LISTINGS_RENTALS = ["Active Listings"]

# Service charge (AED per sqft of built-up area per year)
SERVICE_CHARGE_PER_SQFT = 3.1

# Dubai transaction costs
DLD_FEE_PCT = 0.04  # 4% Dubai Land Department fee
AGENT_COMMISSION_PCT = 0.02  # 2% agent commission
MAINTENANCE_PCT = 0.01  # 1% annual maintenance estimate

# Sub-location ordering
SUBLOC_ORDER = [f"Al Ranim {i}" for i in range(1, 9)]

# Color palette
PALETTE = {
    "primary": "#1B3A5C",
    "secondary": "#2E86AB",
    "accent": "#F18F01",
    "success": "#2BA84A",
    "danger": "#E63946",
    "warning": "#F4A261",
    "light": "#F8F9FA",
    "dark": "#212529",
    # Sub-location colors
    "Al Ranim 1": "#1B3A5C",
    "Al Ranim 2": "#2E86AB",
    "Al Ranim 3": "#2BA84A",
    "Al Ranim 4": "#F18F01",
    "Al Ranim 5": "#E63946",
    "Al Ranim 6": "#9B59B6",
    "Al Ranim 7": "#F4A261",
    "Al Ranim 8": "#1ABC9C",
    # Unit types
    "Townhouse": "#2E86AB",
    "Villa": "#F18F01",
    # Beds
    "3": "#2E86AB",
    "4": "#E63946",
    # Evidence types
    "Active Listings": "#F4A261",
    "Oqood": "#2E86AB",
    "Title Deed": "#2BA84A",
    "Pending Sales (SPA/MOU)": "#9B59B6",
    "Rental Contracts": "#1B3A5C",
}

# Encoding for CSV files
CSV_ENCODING = "latin-1"

# Excluded unit types
EXCLUDED_UNIT_TYPES = ["Apartment"]
