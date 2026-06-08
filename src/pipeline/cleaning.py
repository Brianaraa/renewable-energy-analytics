"""
pipeline/cleaning.py
====================
Production-grade data cleaning pipeline.

Design decisions:
- Returns a new DataFrame (immutable pattern) — no side effects.
- Each step is a pure function → easy to test in isolation.
- Parquet output with pyarrow engine: 10-20× faster than CSV for repeated loads.
- Categorical dtypes for low-cardinality string columns → ~60% memory savings.
"""

import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    FOSSIL_FUELS,
    NUCLEAR_FUELS,
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    RENEWABLE_FUELS,
)

logger = logging.getLogger(__name__)


# ── Public entry point ────────────────────────────────────────────────────────

def run_pipeline(
    raw_path: Path = RAW_DATA_FILE,
    force: bool = False,
) -> pd.DataFrame:
    """
    Load → clean → feature-engineer → save.
    Skips processing if the parquet cache exists (unless force=True).
    """
    out_path = PROCESSED_DATA_FILE.parent / f"{raw_path.stem}_clean.parquet"
    
    if not force and out_path.exists():
        logger.info("Cache hit — loading %s", out_path)
        return pd.read_parquet(out_path)

    logger.info("Starting cleaning pipeline …")
    df = _load_raw(raw_path)
    df = _auto_map_columns(df)
    df = _drop_unusable_rows(df)
    df = _clean_coordinates(df)
    df = _normalize_fuel_types(df)
    df = _clean_capacity(df)
    df = _parse_commissioning_year(df)
    df = _engineer_features(df)
    df = _optimise_dtypes(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)
    logger.info("Saved %d rows → %s", len(df), out_path)
    return df


# ── Step functions ────────────────────────────────────────────────────────────

def _load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download from: https://datasets.wri.org/dataset/globalpowerplantdatabase"
        )
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows, %d cols", *df.shape)
    return df


def _auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Dynamically rename columns to internal expected names using regex."""
    df = df.copy()
    col_map = {}
    
    # Map if the exact lowercase internal name doesn't exist yet
    for expected_col, pattern in [
        ("latitude", r"^(lat|latitude|y)$"),
        ("longitude", r"^(lon|long|longitude|lng|x)$"),
        ("capacity_mw", r"(capacity|cap|mw)"),
        ("primary_fuel", r"(fuel|source|type|category|energy)"),
        ("commissioning_year", r"(year|commission|date|built)"),
        ("country", r"(country|nation|state)")
    ]:
        if expected_col not in df.columns:
            for c in df.columns:
                # Avoid re-mapping if we already decided to map it to something else
                if c not in col_map and re.search(pattern, c, re.IGNORECASE):
                    col_map[c] = expected_col
                    break

    if col_map:
        logger.info("Auto-mapped columns: %s", col_map)
        df = df.rename(columns=col_map)
    return df


def _drop_unusable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing essential columns if they exist in the dataset."""
    before = len(df)
    cols = [c for c in ["latitude", "longitude", "capacity_mw"] if c in df.columns]
    if cols:
        df = df.dropna(subset=cols, how="all")
    logger.info("Dropped %d rows", before - len(df))
    return df.reset_index(drop=True)


def _clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Clamp lat/lon to valid ranges; mark obviously wrong points as NaN."""
    df = df.copy()
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df

    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    invalid_lat = (df["latitude"].abs()  > 90)
    invalid_lon = (df["longitude"].abs() > 180)
    df.loc[invalid_lat, "latitude"]   = np.nan
    df.loc[invalid_lon, "longitude"]  = np.nan

    n_bad = (invalid_lat | invalid_lon).sum()
    if n_bad:
        logger.warning("%d rows have out-of-range coordinates → set to NaN", n_bad)

    return df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def _normalize_fuel_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise primary_fuel strings and derive energy_category.
    Unknown/rare fuels are bucketed as 'Other'.
    """
    df = df.copy()
    if "primary_fuel" not in df.columns:
        return df
        
    known = RENEWABLE_FUELS | FOSSIL_FUELS | NUCLEAR_FUELS

    df["primary_fuel"] = (
        df["primary_fuel"]
        .astype(str)
        .str.strip()
        .str.title()
        .apply(lambda x: x if x in known else "Other")
    )

    def _categorise(fuel: str) -> str:
        if fuel in RENEWABLE_FUELS:   return "Renewable"
        if fuel in FOSSIL_FUELS:      return "Fossil"
        if fuel in NUCLEAR_FUELS:     return "Nuclear"
        return "Other"

    df["energy_category"] = df["primary_fuel"].map(_categorise)
    return df


def _clean_capacity(df: pd.DataFrame) -> pd.DataFrame:
    """Remove physically impossible capacity values (<0 or >100 GW)."""
    df = df.copy()
    if "capacity_mw" not in df.columns:
        return df
    df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce")
    mask = (df["capacity_mw"] > 0) & (df["capacity_mw"] <= 100_000)
    logger.info("Removing %d rows with invalid capacity_mw", (~mask).sum())
    return df[mask].reset_index(drop=True)


def _parse_commissioning_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "commissioning_year" not in df.columns:
        return df
    df["commissioning_year"] = pd.to_numeric(
        df["commissioning_year"], errors="coerce"
    )
    # Sanity-check: between 1900 and current year
    current_year = datetime.now().year
    valid = df["commissioning_year"].between(1900, current_year, inclusive="both")
    df.loc[~valid, "commissioning_year"] = np.nan
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive analytics-ready features.
    All computations are vectorised — no loops.
    """
    df = df.copy()
    GEN_COLS = [
        "generation_gwh_2013", "generation_gwh_2014", "generation_gwh_2015",
        "generation_gwh_2016", "generation_gwh_2017", "generation_gwh_2018",
        "generation_gwh_2019",
    ]
    EST_COLS = [
        "estimated_generation_gwh_2013", "estimated_generation_gwh_2014",
        "estimated_generation_gwh_2015", "estimated_generation_gwh_2016",
        "estimated_generation_gwh_2017",
    ]

    # Coerce to numeric
    for col in GEN_COLS + EST_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Total & average reported generation
    available_gen = [c for c in GEN_COLS if c in df.columns]
    df["total_generation_gwh"] = df[available_gen].sum(axis=1, min_count=1)
    df["avg_generation_gwh"]   = df[available_gen].mean(axis=1)

    # Generation growth (2013 → 2019, % change)
    if "generation_gwh_2013" in df.columns and "generation_gwh_2019" in df.columns:
        df["generation_growth_pct"] = (
            (df["generation_gwh_2019"] - df["generation_gwh_2013"])
            / df["generation_gwh_2013"].replace(0, np.nan)
        ) * 100

    # Plant age (relative to current year)
    current_year = datetime.now().year
    if "commissioning_year" in df.columns:
        df["plant_age_years"] = current_year - df["commissioning_year"]

    # Capacity factor proxy (avg GWh / theoretical max GWh per year)
    if "capacity_mw" in df.columns and "avg_generation_gwh" in df.columns:
        theoretical_max = df["capacity_mw"] * 8.760
        df["capacity_factor_proxy"] = (
            df["avg_generation_gwh"] / theoretical_max.replace(0, np.nan)
        ).clip(0, 1)

    # Sustainability score (0-100): renewable=high, fossil=low
    score_map = {
        "Solar": 95, "Wind": 93, "Hydro": 80,
        "Geothermal": 85, "Biomass": 70, "Wave and Tidal": 90,
        "Storage": 75, "Nuclear": 60,
        "Gas": 30, "Oil": 15, "Coal": 5,
        "Cogeneration": 35, "Petcoke": 10, "Other": 40,
    }
    if "primary_fuel" in df.columns:
        df["sustainability_score"] = df["primary_fuel"].map(score_map).fillna(40)

    # Carbon intensity (kg CO2 / MWh) — IPCC median values
    carbon_map = {
        "Solar": 48, "Wind": 11, "Hydro": 24,
        "Geothermal": 38, "Biomass": 230, "Wave and Tidal": 17,
        "Storage": 0, "Nuclear": 12,
        "Gas": 490, "Oil": 650, "Coal": 820,
        "Cogeneration": 450, "Petcoke": 880, "Other": 500,
    }
    if "primary_fuel" in df.columns:
        df["carbon_intensity_kg_mwh"] = df["primary_fuel"].map(carbon_map).fillna(500)

    # Estimated annual carbon (tonnes CO2)
    if "avg_generation_gwh" in df.columns and "carbon_intensity_kg_mwh" in df.columns:
        df["annual_co2_tonnes"] = (
            df["avg_generation_gwh"] * 1000 * df["carbon_intensity_kg_mwh"] / 1000
        )

    # Is renewable flag
    if "energy_category" in df.columns:
        df["is_renewable"] = df["energy_category"] == "Renewable"

    return df


def _optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns and use Categorical for strings.
    Typically reduces memory by 40-60%.
    """
    df = df.copy()

    # Float64 → float32 for geo and metrics
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    # Object → category for low-cardinality strings
    cat_cols = [
        "country", "primary_fuel", "energy_category",
        "geolocation_source", "generation_data_source",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df
