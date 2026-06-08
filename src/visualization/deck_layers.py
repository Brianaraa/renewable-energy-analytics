"""
visualization/deck_layers.py
=============================
PyDeck / Deck.gl layer builders for 3D visualizations in Streamlit.

Why PyDeck?
- WebGL-powered → renders 100k+ points smoothly in browser.
- Native Streamlit integration via st.pydeck_chart().
- Supports 3D extrusion, heatmaps, arc layers, animated scatter.
"""

import numpy as np
import pandas as pd
import pydeck as pdk

from src.config import FUEL_COLORS, MAP_CENTER_LAT, MAP_CENTER_LON


def _hex_to_rgb(hex_color: str) -> list[int]:
    """Convert '#RRGGBB' → [R, G, B]."""
    h = hex_color.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]


def _prepare_colors(df: pd.DataFrame) -> pd.DataFrame:
    """Add [R,G,B,A] color column based on primary_fuel."""
    df = df.copy()
    df["color"] = df["primary_fuel"].map(
        lambda f: _hex_to_rgb(FUEL_COLORS.get(f, "#888888")) + [200]
    )
    return df


# ── 3D Column / Extrusion Layer ───────────────────────────────────────────────

def column_layer(df: pd.DataFrame, elevation_scale: float = 50.0) -> pdk.Layer:
    """
    3D vertical bars extruded from the map based on capacity_mw.
    elevation_scale: multiplier — tune to taste.
    """
    data = _prepare_colors(df[["latitude", "longitude", "capacity_mw",
                               "primary_fuel", "name", "country"]].dropna())
    data["elevation"] = (data["capacity_mw"] * elevation_scale).clip(100, 5_000_000)

    return pdk.Layer(
        "ColumnLayer",
        data=data,
        get_position=["longitude", "latitude"],
        get_elevation="elevation",
        elevation_scale=1,
        radius=15_000,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
        extruded=True,
    )


# ── Heatmap Layer ─────────────────────────────────────────────────────────────

def heatmap_layer(df: pd.DataFrame, weight_col: str = "capacity_mw") -> pdk.Layer:
    data = df[["latitude", "longitude", weight_col]].dropna().copy()
    data["weight"] = np.log1p(data[weight_col])  # log-scale to avoid dominance

    return pdk.Layer(
        "HeatmapLayer",
        data=data,
        get_position=["longitude", "latitude"],
        get_weight="weight",
        aggregation="MEAN",
        radiusPixels=30,
        intensity=1.5,
        threshold=0.03,
    )


# ── Scatter Plot Layer ────────────────────────────────────────────────────────

def scatterplot_layer(df: pd.DataFrame, radius_scale: float = 800.0) -> pdk.Layer:
    data = _prepare_colors(df[["latitude", "longitude", "capacity_mw",
                               "primary_fuel", "name", "country",
                               "sustainability_score"]].dropna())
    data["radius"] = np.sqrt(data["capacity_mw"]) * radius_scale

    return pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position=["longitude", "latitude"],
        get_radius="radius",
        get_fill_color="color",
        get_line_color=[255, 255, 255, 30],
        line_width_min_pixels=1,
        pickable=True,
        opacity=0.8,
    )


# ── Arc Layer (country → country) ────────────────────────────────────────────

def arc_layer(connections_df: pd.DataFrame) -> pdk.Layer:
    """
    connections_df must have columns:
    src_lat, src_lon, tgt_lat, tgt_lon, capacity_mw
    """
    return pdk.Layer(
        "ArcLayer",
        data=connections_df,
        get_source_position=["src_lon", "src_lat"],
        get_target_position=["tgt_lon", "tgt_lat"],
        get_source_color=[0, 212, 255, 180],
        get_target_color=[0, 255, 159, 180],
        get_width=2,
        pickable=True,
        auto_highlight=True,
    )


# ── Hexagon Layer (density) ───────────────────────────────────────────────────

def hexagon_layer(df: pd.DataFrame, radius: int = 100_000) -> pdk.Layer:
    data = df[["latitude", "longitude", "capacity_mw"]].dropna()

    return pdk.Layer(
        "HexagonLayer",
        data=data,
        get_position=["longitude", "latitude"],
        radius=radius,
        elevation_scale=50,
        elevation_range=[0, 3000],
        get_elevation_weight="capacity_mw",
        extruded=True,
        pickable=True,
        coverage=0.9,
        color_range=[
            [1, 152, 189], [73, 227, 206], [216, 254, 181],
            [254, 237, 177], [254, 173, 84], [209, 55, 78],
        ],
    )


# ── Full 3D Deck ──────────────────────────────────────────────────────────────

def build_3d_deck(
    df: pd.DataFrame,
    layer_type: str = "column",
    pitch: float = 45.0,
    bearing: float = -10.0,
    zoom: float = 1.5,
) -> pdk.Deck:
    """
    Build a complete pdk.Deck ready for st.pydeck_chart().

    layer_type options: 'column' | 'heatmap' | 'scatter' | 'hexagon'
    """
    layer_map = {
        "column":  lambda: column_layer(df),
        "heatmap": lambda: heatmap_layer(df),
        "scatter": lambda: scatterplot_layer(df),
        "hexagon": lambda: hexagon_layer(df),
    }
    layer = layer_map.get(layer_type, layer_map["column"])()

    view = pdk.ViewState(
        latitude=MAP_CENTER_LAT,
        longitude=MAP_CENTER_LON,
        zoom=zoom,
        pitch=pitch,
        bearing=bearing,
    )

    tooltip = {
        "html": (
            "<b>{name}</b><br>"
            "Country: {country}<br>"
            "Fuel: {primary_fuel}<br>"
            "Capacity: {capacity_mw} MW"
        ),
        "style": {
            "backgroundColor": "#1A1F2E",
            "color": "#E0E0E0",
            "border": "1px solid #00D4FF",
            "borderRadius": "6px",
            "padding": "8px",
            "fontSize": "12px",
        },
    }

    return pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/dark-v11",
    )


def build_multi_layer_deck(df: pd.DataFrame) -> pdk.Deck:
    """
    Combine scatter + heatmap into a single deck for rich overlay.
    """
    scatter = scatterplot_layer(df)
    heat    = heatmap_layer(df)

    view = pdk.ViewState(
        latitude=MAP_CENTER_LAT,
        longitude=MAP_CENTER_LON,
        zoom=1.5, pitch=30, bearing=0,
    )

    return pdk.Deck(
        layers=[heat, scatter],
        initial_view_state=view,
        map_style="mapbox://styles/mapbox/dark-v11",
    )
