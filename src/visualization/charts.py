"""
visualization/charts.py
========================
All Plotly figures used in the dashboard.

Conventions:
- Every function returns a go.Figure — composable, testable, reusable.
- Dark theme applied globally via DARK_LAYOUT.
- No Streamlit imports here → pure visualization layer.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.config import FUEL_COLORS, ACCENT_COLOR

# ── Shared dark layout ────────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#E0E0E0"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.1)"),
)


# ── 2D Scatter Map ─────────────────────────────────────────────────────────────
def scatter_map(
    df: pd.DataFrame,
    color_col: str = "primary_fuel",
    size_col: str = "capacity_mw",
    title: str = "Global Power Plants",
    zoom: float = 1.2,
) -> go.Figure:
    """
    Interactive scatter mapbox.
    Size encodes capacity, color encodes fuel type.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=550, title=f"{title} (No Coordinates)")
        return fig
        
    df = df.copy()
    if size_col in df.columns:
        df["size_scaled"] = (df[size_col].clip(1, 5000) / 5000 * 20 + 3).fillna(3)
    else:
        df["size_scaled"] = 3
    
    if color_col not in df.columns:
        color_col = None

    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color=color_col,
        size="size_scaled",
        size_max=20,
        hover_name="name",
        hover_data={
            "country": True,
            "capacity_mw": True,
            "primary_fuel": True,
            "commissioning_year": True,
            "size_scaled": False,
            "latitude": False,
            "longitude": False,
        },
        color_discrete_map=FUEL_COLORS,
        zoom=zoom,
        center={"lat": 20, "lon": 0},
        mapbox_style="carto-darkmatter",
        title=title,
    )
    fig.update_layout(**DARK_LAYOUT, height=550)
    return fig


# ── Choropleth Map ─────────────────────────────────────────────────────────────
def choropleth_map(
    country_df: pd.DataFrame,
    value_col: str = "renewable_share_pct",
    title: str = "Renewable Share by Country (%)",
    color_scale: str = "Viridis",
) -> go.Figure:
    if "country" not in country_df.columns or value_col not in country_df.columns:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=500, title=f"{title} (Data Unavailable)")
        return fig
        
    fig = px.choropleth(
        country_df,
        locations="country",
        locationmode="ISO-3",
        color=value_col,
        hover_name="country_long",
        hover_data={
            "total_plants": True,
            "total_capacity_mw": ":.0f",
            "renewable_share_pct": ":.1f",
        },
        color_continuous_scale=color_scale,
        title=title,
        projection="natural earth",
    )
    fig.update_geos(
        showcoastlines=True, coastlinecolor="#333",
        showland=True, landcolor="#1A1F2E",
        showocean=True, oceancolor="#0D1117",
        showlakes=True, lakecolor="#0D1117",
        showframe=False,
    )
    fig.update_layout(**DARK_LAYOUT, height=500,
                      coloraxis_colorbar=dict(thickness=12, len=0.7))
    return fig


# ── Capacity Bar Chart ─────────────────────────────────────────────────────────
def capacity_bar_chart(
    fuel_df: pd.DataFrame,
    metric: str = "total_capacity_mw",
    title: str = "Total Capacity by Fuel Type (MW)",
) -> go.Figure:
    if "primary_fuel" not in fuel_df.columns or metric not in fuel_df.columns:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=450, title=f"{title} (Data Unavailable)")
        return fig
        
    df = fuel_df.sort_values(metric, ascending=True)
    colors = [FUEL_COLORS.get(f, "#888") for f in df["primary_fuel"]]

    fig = go.Figure(go.Bar(
        x=df[metric],
        y=df["primary_fuel"],
        orientation="h",
        marker_color=colors,
        text=df[metric].apply(lambda x: f"{x:,.0f}"),
        textposition="outside",
    ))
    fig.update_layout(**DARK_LAYOUT, title=title, height=450,
                      xaxis_title=metric, yaxis_title="")
    return fig


# ── Generation Timeseries ──────────────────────────────────────────────────────
def generation_timeseries_chart(ts_df: pd.DataFrame) -> go.Figure:
    """
    Stacked area chart: generation by fuel type over years.
    """
    if not {"year", "generation_gwh", "primary_fuel"}.issubset(ts_df.columns):
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=420, title="Generation Timeseries (Data Unavailable)")
        return fig
        
    fig = px.area(
        ts_df,
        x="year",
        y="generation_gwh",
        color="primary_fuel",
        color_discrete_map=FUEL_COLORS,
        title="Global Power Generation by Fuel Type (GWh)",
        groupnorm="",
    )
    fig.update_layout(**DARK_LAYOUT, height=420,
                      xaxis_title="Year", yaxis_title="GWh")
    return fig


# ── Pie / Donut Charts ─────────────────────────────────────────────────────────
def fuel_donut(df: pd.DataFrame, value_col: str = "capacity_mw") -> go.Figure:
    if "primary_fuel" not in df.columns or value_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=380, title="Capacity Share by Fuel Type (Data Unavailable)")
        return fig
        
    fuel_df = (
        df.groupby("primary_fuel", observed=True)[value_col]
        .sum().reset_index()
        .sort_values(value_col, ascending=False)
    )
    colors = [FUEL_COLORS.get(f, "#888") for f in fuel_df["primary_fuel"]]

    fig = go.Figure(go.Pie(
        labels=fuel_df["primary_fuel"],
        values=fuel_df[value_col],
        hole=0.55,
        marker_colors=colors,
        textinfo="percent+label",
    ))
    fig.update_layout(**DARK_LAYOUT, title="Capacity Share by Fuel Type", height=380)
    return fig


def energy_category_donut(df: pd.DataFrame) -> go.Figure:
    if "energy_category" not in df.columns or "capacity_mw" not in df.columns:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=350, title="Renewable vs Fossil vs Nuclear (Data Unavailable)")
        return fig
        
    cat_df = (
        df.groupby("energy_category", observed=True)["capacity_mw"]
        .sum().reset_index()
    )
    color_map = {"Renewable": "#00FF9F", "Fossil": "#FF6B35", "Nuclear": "#FF69B4", "Other": "#888"}
    colors = [color_map.get(c, "#888") for c in cat_df["energy_category"]]

    fig = go.Figure(go.Pie(
        labels=cat_df["energy_category"],
        values=cat_df["capacity_mw"],
        hole=0.6,
        marker_colors=colors,
    ))
    fig.update_layout(**DARK_LAYOUT, title="Renewable vs Fossil vs Nuclear", height=350)
    return fig


# ── Scatter: Age vs Capacity ───────────────────────────────────────────────────
def age_vs_capacity_scatter(df: pd.DataFrame) -> go.Figure:
    if "plant_age_years" not in df.columns or "capacity_mw" not in df.columns:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=420, title="Plant Age vs Capacity (Data Unavailable)")
        return fig
        
    valid = df.dropna(subset=["plant_age_years", "capacity_mw"])
    if len(valid) == 0:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=420, title="Plant Age vs Capacity (no valid data)")
        return fig
    sample = valid.sample(min(3000, len(valid)), random_state=42)
    fig = px.scatter(
        sample,
        x="plant_age_years",
        y="capacity_mw",
        color="primary_fuel",
        color_discrete_map=FUEL_COLORS,
        size="capacity_mw",
        size_max=18,
        hover_name="name",
        title="Plant Age vs Capacity",
        log_y=True,
        opacity=0.7,
    )
    fig.update_layout(**DARK_LAYOUT, height=420,
                      xaxis_title="Plant Age (years)", yaxis_title="Capacity (MW, log)")
    return fig


# ── Top Countries Bar ──────────────────────────────────────────────────────────
def top_countries_bar(
    country_df: pd.DataFrame,
    metric: str = "total_capacity_mw",
    top_n: int = 20,
) -> go.Figure:
    if "country" not in country_df.columns or metric not in country_df.columns:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=600, title=f"Top {top_n} Countries by {metric} (Data Unavailable)")
        return fig
        
    df = country_df.nlargest(top_n, metric).sort_values(metric)
    fig = go.Figure(go.Bar(
        x=df[metric],
        y=df["country_long"].fillna(df["country"]),
        orientation="h",
        marker=dict(
            color=df[metric],
            colorscale="Viridis",
            showscale=False,
        ),
        text=df[metric].apply(lambda x: f"{x:,.0f}"),
        textposition="outside",
    ))
    fig.update_layout(**DARK_LAYOUT, title=f"Top {top_n} Countries by {metric}",
                      height=600, xaxis_title=metric, yaxis_title="")
    return fig


# ── Heatmap: Capacity by Country × Fuel ───────────────────────────────────────
def country_fuel_heatmap(df: pd.DataFrame, top_n_countries: int = 20) -> go.Figure:
    if not {"country", "primary_fuel", "capacity_mw"}.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=500, title="Capacity Heatmap (Data Unavailable)")
        return fig
        
    top_countries = (
        df.groupby("country", observed=True)["capacity_mw"]
        .sum().nlargest(top_n_countries).index
    )
    pivot = (
        df[df["country"].isin(top_countries)]
        .groupby(["country", "primary_fuel"], observed=True)["capacity_mw"]
        .sum().unstack(fill_value=0)
    )
    fig = px.imshow(
        pivot,
        color_continuous_scale="Viridis",
        title=f"Capacity (MW) — Top {top_n_countries} Countries × Fuel Type",
        aspect="auto",
    )
    fig.update_layout(**DARK_LAYOUT, height=500)
    return fig


# ── Sunburst: Region → Fuel ────────────────────────────────────────────────────
def sustainability_sunburst(df: pd.DataFrame) -> go.Figure:
    if not {"energy_category", "primary_fuel", "capacity_mw", "sustainability_score"}.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=500, title="Energy Mix Sunburst (Data Unavailable)")
        return fig
        
    fig = px.sunburst(
        df,
        path=["energy_category", "primary_fuel"],
        values="capacity_mw",
        color="sustainability_score",
        color_continuous_scale="RdYlGn",
        title="Energy Mix — Sustainability Sunburst",
    )
    fig.update_layout(**DARK_LAYOUT, height=500)
    return fig


# ── Animated Timeline Map ──────────────────────────────────────────────────────
def animated_timeline_map(ts_df: pd.DataFrame) -> go.Figure:
    """
    Animated bar chart race by fuel type over years.
    """
    if not {"year", "generation_gwh", "primary_fuel"}.issubset(ts_df.columns):
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=480, title="Animated Timeline (Data Unavailable)")
        return fig
        
    fig = px.bar(
        ts_df.sort_values(["year", "generation_gwh"], ascending=[True, False]),
        x="primary_fuel",
        y="generation_gwh",
        color="primary_fuel",
        color_discrete_map=FUEL_COLORS,
        animation_frame="year",
        animation_group="primary_fuel",
        range_y=[0, ts_df["generation_gwh"].max() * 1.1],
        title="Power Generation by Fuel Type — Animated Timeline",
    )
    fig.update_layout(**DARK_LAYOUT, height=480,
                      xaxis_title="Fuel Type", yaxis_title="GWh")
    return fig


# ── Advanced Analytics Charts ──────────────────────────────────────────────────

def forecast_line_chart(ts_df: pd.DataFrame) -> go.Figure:
    """
    Line chart with actual (solid) and forecast (dashed) generation data.
    """
    if not {"year", "primary_fuel", "generation_gwh", "is_forecast"}.issubset(ts_df.columns):
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=480, title="Generation Forecast (Data Unavailable)")
        return fig
        
    actual = ts_df[~ts_df["is_forecast"]]
    forecast = ts_df[ts_df["is_forecast"]]

    fig = go.Figure()
    fuels = ts_df["primary_fuel"].unique()
    for fuel in fuels:
        color = FUEL_COLORS.get(fuel, "#888")
        # Actual line
        act = actual[actual["primary_fuel"] == fuel].sort_values("year")
        if len(act) > 0:
            fig.add_trace(go.Scatter(
                x=act["year"], y=act["generation_gwh"],
                mode="lines+markers", name=f"{fuel}",
                line=dict(color=color, width=2),
                marker=dict(size=5),
                legendgroup=fuel,
            ))
        # Forecast line (dashed)
        fct = forecast[forecast["primary_fuel"] == fuel].sort_values("year")
        if len(fct) > 0:
            # Connect forecast to last actual point
            if len(act) > 0:
                bridge = pd.DataFrame([{
                    "year": act["year"].iloc[-1],
                    "generation_gwh": act["generation_gwh"].iloc[-1],
                }])
                fct_plot = pd.concat([bridge, fct], ignore_index=True)
            else:
                fct_plot = fct
            fig.add_trace(go.Scatter(
                x=fct_plot["year"], y=fct_plot["generation_gwh"],
                mode="lines", name=f"{fuel} (forecast)",
                line=dict(color=color, width=2, dash="dash"),
                legendgroup=fuel, showlegend=False,
            ))

    fig.add_vline(x=2019.5, line_dash="dot", line_color="#FFD700",
                  annotation_text="Forecast →", annotation_position="top right")
    fig.update_layout(
        **DARK_LAYOUT, height=480,
        title="Generation Trend & Linear Forecast (GWh)",
        xaxis_title="Year", yaxis_title="GWh",
    )
    return fig


def anomaly_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot highlighting anomalous plants detected by IsolationForest.
    """
    plot_df = df.dropna(subset=["capacity_mw", "avg_generation_gwh"]).copy()
    if len(plot_df) == 0 or "is_anomaly" not in plot_df.columns:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=420, title="Anomaly Detection (no data)")
        return fig

    sample = plot_df.sample(min(5000, len(plot_df)), random_state=42)
    sample["anomaly_label"] = sample["is_anomaly"].map({True: "Anomaly", False: "Normal"})

    fig = px.scatter(
        sample,
        x="capacity_mw",
        y="avg_generation_gwh",
        color="anomaly_label",
        color_discrete_map={"Normal": "#00D4FF", "Anomaly": "#FF4444"},
        size="capacity_mw",
        size_max=14,
        hover_name="name",
        hover_data={"country": True, "primary_fuel": True, "anomaly_label": False},
        title="Anomaly Detection — Capacity vs Generation",
        log_x=True, log_y=True,
        opacity=0.7,
    )
    fig.update_layout(**DARK_LAYOUT, height=450,
                      xaxis_title="Capacity (MW, log)",
                      yaxis_title="Avg Generation (GWh, log)")
    return fig


def transition_score_bar(country_df: pd.DataFrame, top_n: int = 25) -> go.Figure:
    """
    Horizontal bar chart of energy transition scores by country.
    """
    if "transition_score" not in country_df.columns:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=500, title="Transition Score (no data)")
        return fig

    df = country_df.nlargest(top_n, "transition_score").sort_values("transition_score")
    fig = go.Figure(go.Bar(
        x=df["transition_score"],
        y=df["country_long"].fillna(df["country"]),
        orientation="h",
        marker=dict(
            color=df["transition_score"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Score", thickness=12, len=0.7),
        ),
        text=df["transition_score"].apply(lambda x: f"{x:.1f}"),
        textposition="outside",
    ))
    fig.update_layout(
        **DARK_LAYOUT, height=600,
        title=f"Top {top_n} Countries — Energy Transition Score (PCA-based)",
        xaxis_title="Transition Score (0–100)",
        yaxis_title="",
    )
    return fig


def carbon_offset_bar(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing total carbon offset (tonnes CO₂ saved) by fuel type.
    """
    if "carbon_offset_tonnes" not in df.columns:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=420, title="Carbon Offset (no data)")
        return fig

    fuel_offset = (
        df[df["carbon_offset_tonnes"] > 0]
        .groupby("primary_fuel", observed=True)["carbon_offset_tonnes"]
        .sum().reset_index()
        .sort_values("carbon_offset_tonnes", ascending=True)
    )
    colors = [FUEL_COLORS.get(f, "#888") for f in fuel_offset["primary_fuel"]]

    fig = go.Figure(go.Bar(
        x=fuel_offset["carbon_offset_tonnes"],
        y=fuel_offset["primary_fuel"],
        orientation="h",
        marker_color=colors,
        text=fuel_offset["carbon_offset_tonnes"].apply(
            lambda x: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x:,.0f}"
        ),
        textposition="outside",
    ))
    fig.update_layout(
        **DARK_LAYOUT, height=420,
        title="CO₂ Offset by Renewable Fuel Type (tonnes saved vs fossil baseline)",
        xaxis_title="Carbon Offset (tonnes CO₂)",
        yaxis_title="",
    )
    return fig
