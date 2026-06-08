"""
dashboard/app.py
=================
Main Streamlit application entry point.

Run with:
    streamlit run dashboard/app.py

Architecture:
- app.py is thin: it only orchestrates layout and wires data → charts.
- All business logic lives in src/analytics/ and src/visualization/.
- Cached loaders (st.cache_data) prevent re-running the pipeline on every
  widget interaction.
"""

import html as html_lib
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# ── path setup (makes `src` importable when run from repo root) ───────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    PROCESSED_DATA_FILE, RAW_DATA_FILE, RAW_DIR,
    DASHBOARD_TITLE, DASHBOARD_ICON,
)
from src.pipeline.cleaning import run_pipeline
from src.analytics.geospatial import (
    aggregate_by_country, aggregate_by_fuel,
    generation_timeseries, kmeans_spatial_clusters,
    compute_hotspots,
)
from src.analytics.advanced import (
    forecast_generation_trend,
    detect_anomalies,
    compute_energy_transition_score,
    estimate_carbon_offset,
)
from src.visualization.charts import (
    scatter_map, choropleth_map, capacity_bar_chart,
    generation_timeseries_chart, fuel_donut, energy_category_donut,
    age_vs_capacity_scatter, top_countries_bar,
    country_fuel_heatmap, sustainability_sunburst,
    animated_timeline_map,
    forecast_line_chart, anomaly_scatter,
    transition_score_bar, carbon_offset_bar,
)
from src.visualization.deck_layers import build_3d_deck
from src.utils.helpers import (
    inject_css, sidebar_filters, apply_filters,
    render_kpi_row, section_header, metric_card,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon=DASHBOARD_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(show_spinner="⚙️ Loading & cleaning dataset…")
def load_data(file_path: Path) -> pd.DataFrame:
    return run_pipeline(raw_path=file_path)


@st.cache_data(show_spinner=False)
def get_country_agg(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_by_country(df)


@st.cache_data(show_spinner=False)
def get_fuel_agg(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_by_fuel(df)


@st.cache_data(show_spinner=False)
def get_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    return generation_timeseries(df)


@st.cache_data(show_spinner=False)
def get_clustered(df: pd.DataFrame) -> pd.DataFrame:
    return kmeans_spatial_clusters(df)


@st.cache_data(show_spinner="🔮 Running forecast model…")
def get_forecast(ts_df: pd.DataFrame) -> pd.DataFrame:
    return forecast_generation_trend(ts_df)


@st.cache_data(show_spinner="🔍 Detecting anomalies…")
def get_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    return detect_anomalies(df)


@st.cache_data(show_spinner=False)
def get_transition_scores(country_df: pd.DataFrame) -> pd.DataFrame:
    return compute_energy_transition_score(country_df)


@st.cache_data(show_spinner=False)
def get_carbon_offset(df: pd.DataFrame) -> pd.DataFrame:
    return estimate_carbon_offset(df)


# (Deferred to after sidebar file selection)

st.sidebar.markdown(
    '<div style="text-align:center; font-size:2.5rem; padding:8px 0;">⚡</div>',
    unsafe_allow_html=True,
)
st.sidebar.title("⚡ Energy Analytics")
st.sidebar.caption("Global Power Plant Intelligence")

# ── Dynamic File Selector ──
csv_files = list(RAW_DIR.glob("*.csv"))
if not csv_files:
    st.error(f"No CSV files found in {RAW_DIR}")
    st.stop()

default_idx = next((i for i, f in enumerate(csv_files) if f.name == "global_power_plant_database.csv"), 0)

selected_file = st.sidebar.selectbox(
    "📂 Select Dataset",
    options=csv_files,
    format_func=lambda x: x.name,
    index=default_idx
)

# ── Load ──────────────────────────────────────────────────────────────────────
df_full = load_data(selected_file)

filters = sidebar_filters(df_full)
df = apply_filters(df_full, filters)

# warn if very few results
if len(df) == 0:
    st.sidebar.error("❌ No plants match current filters. Adjust filters.")
elif len(df) < 100:
    st.sidebar.warning(f"⚠️ Only {len(df)} plants match current filters.")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 10px 0 0px;">
  <h1 style="font-family:'Space Grotesk',sans-serif; font-size:2.8rem;
             background: linear-gradient(90deg, #00D4FF, #00FF9F);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;
             margin: 0; line-height: 1.2;">
    Global Renewable Energy Analytics
  </h1>
  <p style="color:#9CA3AF; margin:8px 0 16px; font-size:1.05rem; font-weight: 500;">
    Geospatial Intelligence Dashboard for Power Infrastructure
  </p>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ About this Dashboard", expanded=False):
    st.markdown("""
    Welcome to the **Global Renewable Energy Analytics** platform. This portfolio-grade dashboard processes
    and analyzes power plant infrastructure data using **Streamlit, Plotly, PyDeck**, and **Scikit-learn**.
    
    **Key Capabilities:**
    - 🗺️ **Geospatial Intelligence**: K-Means clustering, DBSCAN hotspot detection, and 3D WebGL visualizations.
    - 📈 **Predictive Analytics**: Linear generation forecasting and Isolation Forest anomaly detection.
    - 🔄 **Dynamic Data Pipeline**: Drop any supported CSV into `data/raw/` and the system auto-adapts.
    
    *Built to demonstrate full-stack data engineering, UI/UX design, and machine learning integration.*
    """)
    st.markdown("---")

# ── KPI Row ───────────────────────────────────────────────────────────────────
render_kpi_row(df)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ World Map",
    "📊 Analytics",
    "🌍 Countries",
    "🔥 3D Visualization",
    "📈 Timeseries",
    "🧪 Advanced",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — WORLD MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if len(df) == 0:
        st.info("No data to display. Adjust sidebar filters.")
    else:
        col1, col2 = st.columns([3, 1])

        with col1:
            section_header("Interactive Power Plant Map")
            color_opt = st.selectbox(
                "Color by",
                ["primary_fuel", "energy_category"],
                key="map_color",
            )
            sample = df if len(df) <= 30_000 else df.sample(30_000, random_state=42)
            st.plotly_chart(
                scatter_map(sample, color_col=color_opt, title=""),
                use_container_width=True,
            )

        with col2:
            section_header("Hotspots")
            hotspots = compute_hotspots(df, top_n=10)
            for _, row in hotspots.iterrows():
                safe_name = html_lib.escape(str(row['name'])[:35])
                safe_country = html_lib.escape(str(row['country']))
                safe_fuel = html_lib.escape(str(row['primary_fuel']))
                st.markdown(f"""
                <div style="padding:8px; margin:4px 0; background:#1A1F2E;
                            border-radius:8px; border-left:3px solid #00D4FF;">
                  <div style="font-size:0.75rem; color:#00D4FF;">{safe_fuel}</div>
                  <div style="font-size:0.85rem; font-weight:600;">{safe_name}</div>
                  <div style="font-size:0.75rem; color:#8892A4;">
                    {safe_country} · {row['capacity_mw']:,.0f} MW
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if len(df) == 0:
        st.info("No data to display. Adjust sidebar filters.")
    else:
        fuel_df  = get_fuel_agg(df)

        c1, c2 = st.columns(2)
        with c1:
            section_header("Capacity by Fuel Type")
            st.plotly_chart(capacity_bar_chart(fuel_df), use_container_width=True)
        with c2:
            section_header("Energy Category Split")
            st.plotly_chart(energy_category_donut(df), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            section_header("Fuel Type Capacity Share")
            st.plotly_chart(fuel_donut(df), use_container_width=True)
        with c4:
            section_header("Plant Age vs Capacity")
            st.plotly_chart(age_vs_capacity_scatter(df), use_container_width=True)

        section_header("Sustainability Sunburst")
        st.plotly_chart(sustainability_sunburst(df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COUNTRIES
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if len(df) == 0:
        st.info("No data to display. Adjust sidebar filters.")
    else:
        country_df = get_country_agg(df)

        choropleth_metric = st.selectbox(
            "Choropleth metric",
            ["renewable_share_pct", "total_capacity_mw", "avg_sustainability",
             "total_plants", "total_generation_gwh"],
            key="choro_metric",
        )

        section_header("Global Choropleth Map")
        st.plotly_chart(
            choropleth_map(country_df, value_col=choropleth_metric),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            section_header("Top 20 Countries by Capacity")
            st.plotly_chart(
                top_countries_bar(country_df, metric="total_capacity_mw"),
                use_container_width=True,
            )
        with c2:
            section_header("Top 20 by Renewable Share")
            st.plotly_chart(
                top_countries_bar(country_df, metric="renewable_share_pct"),
                use_container_width=True,
            )

        section_header("Country × Fuel Heatmap")
        st.plotly_chart(country_fuel_heatmap(df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — 3D VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if len(df) == 0:
        st.info("No data to display. Adjust sidebar filters.")
    else:
        section_header("3D WebGL Map (PyDeck)")

        layer_type = filters["map_layer"]
        st.caption(
            f"Layer: **{layer_type.title()}** · "
            f"Showing {min(len(df), 50_000):,} plants"
        )

        sample_3d = df if len(df) <= 50_000 else df.sample(50_000, random_state=42)

        try:
            deck = build_3d_deck(sample_3d, layer_type=layer_type, pitch=45, bearing=-10)
            st.pydeck_chart(deck, use_container_width=True)
        except Exception as e:
            st.error(f"3D render failed: {e}")
            st.info("Tip: Set MAPBOX_TOKEN in .env for best map tiles.")

        st.markdown("""
        <small style="color:#8892A4;">
        🎮 Controls: Left-click drag to pan · Right-click drag to rotate ·
        Scroll to zoom · Hover for tooltip
        </small>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TIMESERIES
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    if len(df) == 0:
        st.info("No data to display. Adjust sidebar filters.")
    else:
        ts_df = get_timeseries(df)

        section_header("Generation by Fuel Type Over Time")
        st.plotly_chart(generation_timeseries_chart(ts_df), use_container_width=True)

        section_header("Animated Bar Chart Race")
        st.plotly_chart(animated_timeline_map(ts_df), use_container_width=True)

        # Raw data explorer
        section_header("🔍 Data Explorer")
        with st.expander("Browse filtered dataset"):
            cols_show = [
                "name", "country", "primary_fuel", "energy_category",
                "capacity_mw", "commissioning_year", "sustainability_score",
                "total_generation_gwh", "latitude", "longitude",
            ]
            st.dataframe(
                df[[c for c in cols_show if c in df.columns]].head(500),
                use_container_width=True,
                height=350,
            )
            st.caption(f"Showing first 500 of {len(df):,} filtered plants.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ADVANCED ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    if len(df) == 0:
        st.info("No data to display. Adjust sidebar filters.")
    else:
        st.markdown("""
        <div style="padding:12px 16px; background:#1A1F2E; border-radius:10px;
                    border-left:4px solid #9370DB; margin-bottom:20px;">
          <div style="font-size:0.85rem; color:#E0E0E0;">
            🧪 <strong>Advanced Analytics</strong> — ML-powered insights including
            linear trend forecasting, IsolationForest anomaly detection,
            PCA-based energy transition scoring, and carbon offset estimation.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 1. Generation Forecast ────────────────────────────────────────
        section_header("🔮 Generation Trend Forecast (to 2025)")
        ts_df = get_timeseries(df)
        ts_with_forecast = get_forecast(ts_df)

        st.plotly_chart(forecast_line_chart(ts_with_forecast), use_container_width=True)

        with st.expander("ℹ️ About this forecast"):
            st.markdown("""
            - **Method**: Linear regression per fuel type on 2013–2019 data
            - **Forecast horizon**: 2020–2025
            - **Limitation**: Simple linear extrapolation — does not capture
              policy changes, COVID impact, or exponential growth in renewables
            - Dashed lines represent forecast; solid lines are actuals
            """)

        st.markdown("---")

        # ── 2. Anomaly Detection ──────────────────────────────────────────
        section_header("🔍 Anomaly Detection (IsolationForest)")
        df_anomaly = get_anomalies(df)
        n_anomalies = df_anomaly["is_anomaly"].sum()
        n_total = len(df_anomaly)

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Total Plants Analyzed", f"{n_total:,}", accent="#00D4FF")
        with c2:
            metric_card("Anomalies Detected", f"{n_anomalies:,}", accent="#FF4444")
        with c3:
            pct = (n_anomalies / n_total * 100) if n_total > 0 else 0
            metric_card("Anomaly Rate", f"{pct:.1f}%", accent="#FFD700")

        st.plotly_chart(anomaly_scatter(df_anomaly), use_container_width=True)

        with st.expander("🔎 View Anomalous Plants"):
            anomalous = df_anomaly[df_anomaly["is_anomaly"] == True]
            if len(anomalous) > 0:
                show_cols = ["name", "country", "primary_fuel", "capacity_mw",
                             "avg_generation_gwh", "capacity_factor_proxy",
                             "plant_age_years"]
                st.dataframe(
                    anomalous[[c for c in show_cols if c in anomalous.columns]]
                    .head(100).reset_index(drop=True),
                    use_container_width=True,
                    height=300,
                )
                st.caption(f"Showing top 100 of {len(anomalous):,} anomalous plants.")
            else:
                st.info("No anomalies detected with current filters.")

        st.markdown("---")

        # ── 3. Energy Transition Score ────────────────────────────────────
        section_header("🌱 Energy Transition Score (PCA-based)")
        country_df = get_country_agg(df)
        country_scored = get_transition_scores(country_df)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(
                transition_score_bar(country_scored, top_n=25),
                use_container_width=True,
            )
        with c2:
            st.markdown("""
            <div style="padding:16px; background:#1A1F2E; border-radius:10px;
                        margin-top:40px;">
              <div style="font-size:0.85rem; color:#E0E0E0;">
                <strong>How it works</strong><br><br>
                The transition score combines three factors
                using PCA (Principal Component Analysis):<br><br>
                📊 <strong>Renewable share %</strong><br>
                🌿 <strong>Avg sustainability score</strong><br>
                ⚡ <strong>Avg capacity per plant</strong><br><br>
                Scores are normalised to 0–100.
                Higher = further along the energy transition.
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── 4. Carbon Offset ──────────────────────────────────────────────
        section_header("🌍 Carbon Offset Estimation")
        df_offset = get_carbon_offset(df)

        total_offset = df_offset["carbon_offset_tonnes"].sum()
        total_co2 = df_offset["annual_co2_tonnes"].sum() if "annual_co2_tonnes" in df_offset.columns else 0
        renewable_savers = df_offset["is_carbon_saver"].sum() if "is_carbon_saver" in df_offset.columns else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card(
                "Total CO₂ Offset",
                f"{total_offset/1e6:.1f}M tonnes",
                accent="#00FF9F",
            )
        with c2:
            metric_card(
                "Total CO₂ Emissions",
                f"{total_co2/1e6:.1f}M tonnes",
                accent="#FF6B35",
            )
        with c3:
            metric_card(
                "Carbon-saving Plants",
                f"{renewable_savers:,}",
                accent="#32CD32",
            )

        st.plotly_chart(carbon_offset_bar(df_offset), use_container_width=True)

        with st.expander("ℹ️ About carbon offset estimation"):
            st.markdown("""
            - **Baseline**: Global average fossil mix = 550 kg CO₂/MWh
            - **Method**: For each plant, offset = avg_generation × (550 − plant's
              carbon intensity)
            - **Carbon intensities**: IPCC median lifecycle values per fuel type
            - Only plants with carbon intensity below 550 kg/MWh contribute
              positively to offset
            """)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8892A4; font-size:0.8rem; padding:16px 0;">
  Built with ⚡ Python · Streamlit · Plotly · PyDeck &nbsp;|&nbsp;
  Data: <a href="https://datasets.wri.org/dataset/globalpowerplantdatabase"
           style="color:#00D4FF;">Global Power Plant Database (WRI)</a>
</div>
""", unsafe_allow_html=True)
