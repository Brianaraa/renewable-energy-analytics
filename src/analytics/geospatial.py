"""
analytics/geospatial.py
========================
Spatial analysis helpers: clustering, aggregation, density, hotspots.

Design:
- All functions accept a plain pandas DataFrame (no GeoPandas dependency at
  call-site) — keeps the API portable and testable.
- Heavy operations (K-Means, DBSCAN) are lazy — only run when called.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

from src.config import CLUSTER_N, CLUSTER_RANDOM_STATE


# ── Clustering ────────────────────────────────────────────────────────────────

def kmeans_spatial_clusters(
    df: pd.DataFrame,
    n_clusters: int = CLUSTER_N,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add a 'cluster' column via K-Means on lat/lon (+ optional features).

    Why K-Means over DBSCAN here?
    - Deterministic, fast, stable cluster count.
    - Good for choropleth-style summary cards.
    DBSCAN is better for density-based discovery (see below).
    """
    df = df.copy()
    base = ["latitude", "longitude"]
    cols = base + (features or [])
    valid = df[cols].dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(valid)

    km = KMeans(n_clusters=n_clusters, random_state=CLUSTER_RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(X)

    df.loc[valid.index, "cluster"] = labels
    df["cluster"] = df["cluster"].fillna(-1).astype(int)
    return df


def dbscan_density_clusters(
    df: pd.DataFrame,
    eps_deg: float = 2.0,
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    DBSCAN clustering on lat/lon.
    - eps_deg: neighbourhood radius in degrees (~220 km at equator).
    - Noise points → cluster = -1.

    Use-case: finding natural energy hotspots without pre-specifying count.
    """
    df = df.copy()
    coords = df[["latitude", "longitude"]].dropna().values

    # Convert degrees to radians for haversine metric
    coords_rad = np.radians(coords)
    eps_rad = eps_deg * np.pi / 180

    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine", n_jobs=-1)
    labels = db.fit_predict(coords_rad)

    valid_idx = df[["latitude", "longitude"]].dropna().index
    df.loc[valid_idx, "density_cluster"] = labels
    df["density_cluster"] = df["density_cluster"].fillna(-1).astype(int)
    return df


# ── Country aggregation ───────────────────────────────────────────────────────

def aggregate_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Country-level summary for choropleth maps.
    Returns one row per country.
    """
    if "country" not in df.columns:
        return pd.DataFrame(columns=["country", "total_capacity_mw", "renewable_share_pct"])

    grp = df.groupby("country", observed=True)

    agg = grp.agg(
        country_long=("country_long", "first"),
        total_plants=("gppd_idnr", "count"),
        total_capacity_mw=("capacity_mw", "sum"),
        avg_capacity_mw=("capacity_mw", "mean"),
        total_generation_gwh=("total_generation_gwh", "sum"),
        renewable_plants=("is_renewable", "sum"),
        avg_sustainability=("sustainability_score", "mean"),
        total_co2_tonnes=("annual_co2_tonnes", "sum"),
        avg_plant_age=("plant_age_years", "mean"),
    ).reset_index()

    agg["renewable_share_pct"] = (
        agg["renewable_plants"] / agg["total_plants"] * 100
    ).round(1)

    return agg


def aggregate_by_fuel(df: pd.DataFrame) -> pd.DataFrame:
    """Fuel-type summary for bar/pie charts."""
    if "primary_fuel" not in df.columns:
        return pd.DataFrame(columns=["primary_fuel", "total_capacity_mw"])
        
    return (
        df.groupby("primary_fuel", observed=True)
        .agg(
            plant_count=("gppd_idnr", "count"),
            total_capacity_mw=("capacity_mw", "sum"),
            total_generation_gwh=("total_generation_gwh", "sum"),
            avg_sustainability=("sustainability_score", "mean"),
        )
        .reset_index()
        .sort_values("total_capacity_mw", ascending=False)
    )


def aggregate_by_region_grid(
    df: pd.DataFrame,
    lat_bins: int = 36,
    lon_bins: int = 72,
) -> pd.DataFrame:
    """
    Hexagonal-style grid aggregation (using rectangular bins).
    Returns a dataframe suitable for heatmap layers.

    lat_bins=36  → 5° resolution
    lon_bins=72  → 5° resolution
    """
    df = df.copy()
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return pd.DataFrame()
        
    df["lat_bin"] = pd.cut(df["latitude"],  bins=lat_bins)
    df["lon_bin"] = pd.cut(df["longitude"], bins=lon_bins)

    grid = (
        df.groupby(["lat_bin", "lon_bin"], observed=True)
        .agg(
            plant_count=("gppd_idnr", "count"),
            total_capacity_mw=("capacity_mw", "sum"),
            lat_center=("latitude", "mean"),
            lon_center=("longitude", "mean"),
        )
        .reset_index(drop=True)
        .dropna(subset=["lat_center", "lon_center"])
    )
    return grid


# ── Temporal analysis ─────────────────────────────────────────────────────────

def generation_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a long-format timeseries:
      year | primary_fuel | energy_category | total_gwh
    Suitable for animated timeline charts.
    """
    gen_map = {
        "generation_gwh_2013": 2013, "generation_gwh_2014": 2014,
        "generation_gwh_2015": 2015, "generation_gwh_2016": 2016,
        "generation_gwh_2017": 2017, "generation_gwh_2018": 2018,
        "generation_gwh_2019": 2019,
    }
    available = {k: v for k, v in gen_map.items() if k in df.columns}

    if not available or "primary_fuel" not in df.columns or "energy_category" not in df.columns:
        return pd.DataFrame(columns=["year", "primary_fuel", "energy_category", "generation_gwh"])

    rows = []
    for col, year in available.items():
        sub = df[["primary_fuel", "energy_category", col]].copy()
        sub = sub.rename(columns={col: "generation_gwh"})
        sub["year"] = year
        rows.append(sub)

    ts = pd.concat(rows, ignore_index=True)
    ts = (
        ts.groupby(["year", "primary_fuel", "energy_category"], observed=True)
        ["generation_gwh"]
        .sum()
        .reset_index()
    )
    return ts


# ── Hotspot analysis ──────────────────────────────────────────────────────────

def compute_hotspots(
    df: pd.DataFrame,
    top_n: int = 20,
    metric: str = "capacity_mw",
) -> pd.DataFrame:
    """
    Return the top-N plants by a given metric — highlighted as 'hotspots'.
    """
    if metric not in df.columns:
        return pd.DataFrame()
        
    cols = [c for c in [
        "name", "country", "primary_fuel", "energy_category",
        "latitude", "longitude", "capacity_mw",
        "total_generation_gwh", "sustainability_score",
    ] if c in df.columns]
    
    return (
        df.nlargest(top_n, metric)
        [cols]
        .reset_index(drop=True)
    )


# ── Nearest-neighbour distance (optional) ─────────────────────────────────────

def nearest_neighbour_distance(df: pd.DataFrame, sample_n: int = 5000) -> pd.Series:
    """
    Approximate average nearest-neighbour distance (degrees) for a sample.
    Useful for density analysis narrative.
    """
    from sklearn.neighbors import BallTree

    coords = df[["latitude", "longitude"]].dropna()
    if len(coords) > sample_n:
        coords = coords.sample(sample_n, random_state=42)

    coords_rad = np.radians(coords.values)
    tree = BallTree(coords_rad, metric="haversine")
    distances, _ = tree.query(coords_rad, k=2)  # k=2: exclude self
    dist_deg = np.degrees(distances[:, 1])
    return pd.Series(dist_deg, name="nn_dist_deg")
