"""
config.py - Central configuration for the project.
All constants, paths, and settings live here to avoid magic strings.
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ASSETS_DIR  = ROOT_DIR / "assets"

# ── Dataset ──────────────────────────────────────────────────────────────────
RAW_DATA_FILE       = RAW_DIR / "global_power_plant_database.csv"
PROCESSED_DATA_FILE = PROCESSED_DIR / "plants_clean.parquet"
COUNTRY_GEO_FILE    = PROCESSED_DIR / "countries.geojson"

# ── Fuel Classification ───────────────────────────────────────────────────────
RENEWABLE_FUELS = {
    "Solar", "Wind", "Hydro", "Geothermal",
    "Biomass", "Wave and Tidal", "Storage",
}
FOSSIL_FUELS = {
    "Gas", "Oil", "Coal", "Petcoke", "Cogeneration",
}
NUCLEAR_FUELS = {"Nuclear"}

FUEL_COLORS = {
    "Solar":         "#FFD700",
    "Wind":          "#00BFFF",
    "Hydro":         "#1E90FF",
    "Geothermal":    "#FF6347",
    "Biomass":       "#32CD32",
    "Wave and Tidal":"#40E0D0",
    "Storage":       "#9370DB",
    "Gas":           "#FFA500",
    "Oil":           "#8B4513",
    "Coal":          "#696969",
    "Nuclear":       "#FF69B4",
    "Other":         "#A9A9A9",
}

# ── Map Defaults ──────────────────────────────────────────────────────────────
MAP_CENTER_LAT  = 20.0
MAP_CENTER_LON  = 0.0
MAP_ZOOM        = 1.5
MAPBOX_STYLE    = "mapbox://styles/mapbox/dark-v11"
MAPBOX_FALLBACK = "carto-darkmatter"   # used when no Mapbox token

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_TITLE     = "⚡ Global Renewable Energy Analytics"
DASHBOARD_ICON      = "⚡"
DARK_BG             = "#0E1117"
CARD_BG             = "#1A1F2E"
ACCENT_COLOR        = "#00D4FF"
SUCCESS_COLOR       = "#00FF9F"
WARNING_COLOR       = "#FFD700"

# ── Analytics ─────────────────────────────────────────────────────────────────
GENERATION_COLS = [
    "generation_gwh_2013", "generation_gwh_2014", "generation_gwh_2015",
    "generation_gwh_2016", "generation_gwh_2017", "generation_gwh_2018",
    "generation_gwh_2019",
]
ESTIMATED_GEN_COLS = [
    "estimated_generation_gwh_2013", "estimated_generation_gwh_2014",
    "estimated_generation_gwh_2015", "estimated_generation_gwh_2016",
    "estimated_generation_gwh_2017",
]
YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019]

# ── Clustering ────────────────────────────────────────────────────────────────
CLUSTER_N = 8
CLUSTER_RANDOM_STATE = 42
