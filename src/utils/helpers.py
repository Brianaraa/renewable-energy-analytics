"""
utils/helpers.py
=================
Streamlit UI helper components: metric cards, CSS injection, sidebar filters.
"""

import html

import streamlit as st
import pandas as pd
from src.config import FUEL_COLORS, RENEWABLE_FUELS


# ── Global CSS ────────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}

/* Premium Glassmorphism Card Style */
.metric-card {
    background: rgba(26, 31, 46, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 24px;
    margin: 8px 0;
    position: relative;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}
.metric-card:hover {
    transform: translateY(-5px);
    border-color: rgba(0, 212, 255, 0.4);
    box-shadow: 0 8px 30px rgba(0, 212, 255, 0.15);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: var(--accent, #00D4FF);
    border-radius: 16px 0 0 16px;
    box-shadow: 0 0 10px var(--accent, #00D4FF);
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent, #00D4FF);
    line-height: 1.1;
    margin: 8px 0 4px;
    text-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
}
.metric-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-delta {
    font-size: 0.85rem;
    color: #10B981;
    margin-top: 6px;
    font-weight: 500;
}

/* Section headers */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: #F3F4F6;
    border-left: 4px solid #00D4FF;
    padding-left: 14px;
    margin: 32px 0 20px;
    letter-spacing: 0.02em;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: rgba(13, 17, 23, 0.95) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}

/* Map container shadow */
.deck-container {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255,255,255,0.05);
}

/* Elegant Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    padding-bottom: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(26, 31, 46, 0.4);
    border-radius: 8px;
    color: #9CA3AF;
    padding: 10px 24px;
    font-weight: 500;
    border: 1px solid transparent;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #F3F4F6;
    background: rgba(26, 31, 46, 0.8);
}
.stTabs [aria-selected="true"] {
    background: rgba(0, 212, 255, 0.1);
    color: #00D4FF;
    border: 1px solid rgba(0, 212, 255, 0.3);
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.1);
}
</style>
"""


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: str = "", accent: str = "#00D4FF"):
    st.markdown(f"""
    <div class="metric-card" style="--accent: {accent};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {"<div class='metric-delta'>▲ " + delta + "</div>" if delta else ""}
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ── Sidebar Filters ───────────────────────────────────────────────────────────

def sidebar_filters(df: pd.DataFrame) -> dict:
    """
    Render sidebar filter widgets and return filter state dict.
    All filters return lists for consistent downstream handling.
    """
    st.sidebar.markdown("## ⚙️ Filters")
    st.sidebar.markdown("---")

    # Energy category
    categories = sorted(df["energy_category"].dropna().unique().tolist())
    selected_cats = st.sidebar.multiselect(
        "Energy Category",
        options=categories,
        default=categories,
        help="Filter by Renewable, Fossil, Nuclear, Other",
    )

    # Fuel type
    all_fuels = sorted(df["primary_fuel"].dropna().unique().tolist())
    selected_fuels = st.sidebar.multiselect(
        "Fuel Type",
        options=all_fuels,
        default=all_fuels,
    )

    # Capacity range
    cap_min = float(df["capacity_mw"].min())
    cap_max = float(df["capacity_mw"].max())
    cap_range = st.sidebar.slider(
        "Capacity Range (MW)",
        min_value=cap_min,
        max_value=min(cap_max, 50_000.0),
        value=(cap_min, min(cap_max, 50_000.0)),
        step=10.0,
    )

    # Country (top 30 by plant count)
    top_countries = (
        df["country"].value_counts().head(30).index.tolist()
    )
    all_countries = ["All"] + sorted(df["country"].dropna().unique().tolist())
    selected_country = st.sidebar.selectbox("Country Focus", all_countries)

    # Year range (commissioning)
    min_year = int(df["commissioning_year"].dropna().min())
    max_year = int(df["commissioning_year"].dropna().max())
    year_range = st.sidebar.slider(
        "Commissioning Year",
        min_value=min_year, max_value=max_year,
        value=(max(min_year, 1950), max_year),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗺️ Map Options")
    map_layer = st.sidebar.radio(
        "3D Layer Type",
        ["column", "heatmap", "scatter", "hexagon"],
        format_func=lambda x: x.title(),
    )

    return {
        "categories": selected_cats,
        "fuels": selected_fuels,
        "cap_range": cap_range,
        "country": selected_country,
        "year_range": year_range,
        "map_layer": map_layer,
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar filter state to the dataframe."""
    mask = pd.Series(True, index=df.index)

    if filters["categories"]:
        mask &= df["energy_category"].isin(filters["categories"])
    if filters["fuels"]:
        mask &= df["primary_fuel"].isin(filters["fuels"])

    lo, hi = filters["cap_range"]
    mask &= df["capacity_mw"].between(lo, hi)

    if filters["country"] != "All":
        mask &= df["country"] == filters["country"]

    y_lo, y_hi = filters["year_range"]
    year_mask = (
        df["commissioning_year"].isna() |
        df["commissioning_year"].between(y_lo, y_hi)
    )
    mask &= year_mask

    return df[mask].reset_index(drop=True)


# ── KPI Row ───────────────────────────────────────────────────────────────────

def render_kpi_row(df: pd.DataFrame):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Total Plants", f"{len(df):,}", accent="#00D4FF")
    with c2:
        total_cap = df["capacity_mw"].sum()
        metric_card("Total Capacity", f"{total_cap/1e6:.2f} TW", accent="#00FF9F")
    with c3:
        ren_pct = df["is_renewable"].mean() * 100
        metric_card("Renewable Share", f"{ren_pct:.1f}%", accent="#FFD700")
    with c4:
        countries = df["country"].nunique()
        metric_card("Countries", f"{countries}", accent="#FF6B35")
    with c5:
        avg_sus = df["sustainability_score"].mean()
        metric_card("Avg Sustainability", f"{avg_sus:.0f}/100", accent="#9370DB")
