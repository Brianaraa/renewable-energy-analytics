"""
tests/test_pipeline.py
=======================
Unit tests for the cleaning pipeline and analytics helpers.
Run: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.cleaning import (
    _clean_coordinates,
    _normalize_fuel_types,
    _clean_capacity,
    _engineer_features,
)
from src.analytics.geospatial import (
    aggregate_by_country,
    aggregate_by_fuel,
    generation_timeseries,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "gppd_idnr": [f"P{i}" for i in range(10)],
        "name": [f"Plant {i}" for i in range(10)],
        "country": ["USA"] * 5 + ["DEU"] * 5,
        "country_long": ["United States"] * 5 + ["Germany"] * 5,
        "latitude":  [40.0, 35.0, -200.0, np.nan, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
        "longitude": [-74.0, -118.0, 200.0, np.nan, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        "capacity_mw": [500, 1000, 200, 300, -5, 50, 750, 1200, 80, 0],
        "primary_fuel": ["Solar", "Wind", "Coal", "Gas", "Hydro",
                         "Nuclear", "Oil", "UNKNOWN_FUEL", "Biomass", "Solar"],
        "commissioning_year": [2010, 2015, 1990, 2005, 2020,
                               1985, 2000, 2018, 2012, 1800],
        "generation_gwh_2013": [100, 200, np.nan, 50, 30, 400, 10, 5, 80, 90],
        "generation_gwh_2019": [120, 250, np.nan, 45, 35, 420, 8, 7, 95, 100],
    })


# ── Coordinate cleaning ───────────────────────────────────────────────────────

def test_clean_coordinates_removes_invalid(sample_df):
    clean = _clean_coordinates(sample_df)
    assert clean["latitude"].between(-90, 90).all()
    assert clean["longitude"].between(-180, 180).all()
    assert len(clean) < len(sample_df)


# ── Fuel normalisation ────────────────────────────────────────────────────────

def test_normalize_fuel_unknown_becomes_other(sample_df):
    norm = _normalize_fuel_types(sample_df)
    assert "UNKNOWN_FUEL" not in norm["primary_fuel"].values
    assert "Other" in norm["primary_fuel"].values


def test_energy_category_created(sample_df):
    norm = _normalize_fuel_types(sample_df)
    assert "energy_category" in norm.columns
    assert set(norm["energy_category"].unique()).issubset(
        {"Renewable", "Fossil", "Nuclear", "Other"}
    )


# ── Capacity cleaning ─────────────────────────────────────────────────────────

def test_clean_capacity_removes_negatives(sample_df):
    clean = _clean_capacity(sample_df)
    assert (clean["capacity_mw"] > 0).all()


# ── Feature engineering ───────────────────────────────────────────────────────

def test_engineer_features_adds_columns(sample_df):
    df = _normalize_fuel_types(sample_df)
    df = _clean_capacity(df)
    df = _clean_coordinates(df)
    featured = _engineer_features(df)

    for col in ["total_generation_gwh", "sustainability_score",
                "carbon_intensity_kg_mwh", "is_renewable"]:
        assert col in featured.columns, f"Missing column: {col}"


def test_sustainability_score_range(sample_df):
    df = _normalize_fuel_types(sample_df)
    featured = _engineer_features(df)
    scores = featured["sustainability_score"].dropna()
    assert scores.between(0, 100).all()


# ── Analytics ─────────────────────────────────────────────────────────────────

def test_aggregate_by_country(sample_df):
    df = _normalize_fuel_types(sample_df)
    df = _clean_capacity(df)
    df = _clean_coordinates(df)
    df = _engineer_features(df)
    df["is_renewable"] = df["energy_category"] == "Renewable"

    country_agg = aggregate_by_country(df)
    assert "country" in country_agg.columns
    assert "renewable_share_pct" in country_agg.columns
    assert (country_agg["renewable_share_pct"].between(0, 100)).all()


def test_generation_timeseries_long_format(sample_df):
    df = _normalize_fuel_types(sample_df)
    df = _clean_capacity(df)
    df = _clean_coordinates(df)
    df = _engineer_features(df)

    ts = generation_timeseries(df)
    assert "year" in ts.columns
    assert "generation_gwh" in ts.columns
    assert ts["year"].nunique() >= 2

# ── Dynamic Mapping & Fallbacks ───────────────────────────────────────────────

from src.pipeline.cleaning import _auto_map_columns

def test_auto_map_columns():
    df = pd.DataFrame({
        "Lat": [1.0, 2.0],
        "Long": [10.0, 20.0],
        "capacity_MW": [100, 200],
        "fuel1": ["Solar", "Wind"],
        "Nation": ["USA", "CAN"]
    })
    mapped_df = _auto_map_columns(df.copy())
    assert "latitude" in mapped_df.columns
    assert "longitude" in mapped_df.columns
    assert "capacity_mw" in mapped_df.columns
    assert "primary_fuel" in mapped_df.columns
    assert "country" in mapped_df.columns

def test_aggregate_by_country_empty():
    empty_df = pd.DataFrame(columns=[
        "country", "capacity_mw", "is_renewable", "sustainability_score", "total_generation_gwh",
        "country_long", "gppd_idnr", "annual_co2_tonnes", "plant_age_years"
    ])
    agg_df = aggregate_by_country(empty_df)
    assert len(agg_df) == 0
    assert "total_capacity_mw" in agg_df.columns
