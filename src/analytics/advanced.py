"""
analytics/advanced.py
======================
Advanced analytics: trend forecasting, anomaly detection, ML scoring.

These are optional enrichments — the dashboard works without them.
They're designed to be run as a preprocessing step and stored in parquet.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ── Linear Trend Forecasting ─────────────────────────────────────────────────

def forecast_generation_trend(
    ts_df: pd.DataFrame,
    forecast_years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Simple linear regression per fuel type → forecast to 2025.
    Returns ts_df extended with forecast rows (is_forecast=True).
    """
    from numpy.polynomial import polynomial as P

    if not {"primary_fuel", "year", "generation_gwh"}.issubset(ts_df.columns):
        ts_actual = ts_df.copy()
        ts_actual["is_forecast"] = False
        return ts_actual

    if forecast_years is None:
        forecast_years = [2020, 2021, 2022, 2023, 2024, 2025]

    result_rows = []

    for fuel, group in ts_df.groupby("primary_fuel"):
        group_sorted = group.sort_values("year")
        x = group_sorted["year"].values.astype(float)
        y = group_sorted["generation_gwh"].values

        if len(x) < 3:
            continue

        # Fit degree-1 polynomial (robust to outliers vs np.polyfit)
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs

        for yr in forecast_years:
            pred = max(0, slope * yr + intercept)
            result_rows.append({
                "year": yr,
                "primary_fuel": fuel,
                "energy_category": group["energy_category"].iloc[0],
                "generation_gwh": pred,
                "is_forecast": True,
            })

    ts_actual = ts_df.copy()
    ts_actual["is_forecast"] = False

    if result_rows:
        ts_forecast = pd.DataFrame(result_rows)
        return pd.concat([ts_actual, ts_forecast], ignore_index=True)
    return ts_actual


# ── Anomaly Detection ─────────────────────────────────────────────────────────

def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.03,
) -> pd.DataFrame:
    """
    IsolationForest anomaly detection on operational features.
    Marks ~3% of plants as anomalies (likely data quality issues or
    genuinely unusual plants).

    Features used:
    - capacity_mw
    - avg_generation_gwh
    - capacity_factor_proxy
    - plant_age_years
    """
    df = df.copy()
    features = ["capacity_mw", "avg_generation_gwh", "capacity_factor_proxy",
                "plant_age_years"]
    available = [f for f in features if f in df.columns]

    if not available:
        df["is_anomaly"] = False
        return df

    valid = df[available].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(valid)

    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    labels = iso.fit_predict(X)

    df.loc[valid.index, "is_anomaly"] = labels == -1
    df["is_anomaly"] = df["is_anomaly"].fillna(False)
    return df


# ── PCA Energy Transition Score ───────────────────────────────────────────────

def compute_energy_transition_score(country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-dimensional energy transition score using PCA-weighted composite.
    Factors: renewable_share, avg_sustainability, renewable/total ratio,
    capacity per plant (efficiency proxy).

    Returns country_df with added 'transition_score' column (0-100).
    """
    df = country_df.copy()
    features = [
        "renewable_share_pct",
        "avg_sustainability",
        "avg_capacity_mw",
    ]
    available = [f for f in features if f in df.columns]
    valid = df[available].dropna()

    if len(valid) < 5:
        df["transition_score"] = 50.0
        return df

    scaler = StandardScaler()
    X = scaler.fit_transform(valid)

    # Use PCA first component as a composite score
    pca = PCA(n_components=1)
    scores_raw = pca.fit_transform(X).flatten()

    # Normalise to 0-100
    s_min, s_max = scores_raw.min(), scores_raw.max()
    scores_norm = (scores_raw - s_min) / (s_max - s_min + 1e-8) * 100

    df.loc[valid.index, "transition_score"] = scores_norm
    df["transition_score"] = df["transition_score"].fillna(50.0)
    return df


# ── Carbon Offset Estimation ──────────────────────────────────────────────────

def estimate_carbon_offset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate how much CO2 would be emitted if each renewable plant
    were replaced by the global average fossil mix (~550 kg CO2/MWh).

    carbon_offset_tonnes = avg_generation_gwh * 1000 *
                           (550 - carbon_intensity_kg_mwh) / 1000
    """
    df = df.copy()
    FOSSIL_BASELINE = 550  # kg CO2/MWh

    if "avg_generation_gwh" not in df.columns or "carbon_intensity_kg_mwh" not in df.columns:
        df["carbon_offset_tonnes"] = 0
        df["is_carbon_saver"] = False
        return df

    df["carbon_offset_tonnes"] = (
        df["avg_generation_gwh"] * 1000
        * (FOSSIL_BASELINE - df["carbon_intensity_kg_mwh"].fillna(0))
        / 1000
    ).clip(lower=0)

    df["is_carbon_saver"] = df["carbon_offset_tonnes"] > 0
    return df
