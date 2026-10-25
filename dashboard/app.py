"""
Weather and Road Collisions in Great Britain, 2025: interactive dashboard.


Inputs (produced by notebooks 03 and 04):
    data/merged/collisions_weather_2025.csv            one row per collision, with matched weather
    data/merged/daily_gb_weather_collisions_2025.csv   one row per day, GB weather and collisions
    dashboard/exposure_counts.npz                      days each 5 km cell spent in each weather band,
                                                       by month and weekday (notebook 04, Step 8)

Method (same as notebook 04):
    Frequency: collisions observed vs expected for the same 5 km cells, months and
               weekdays, by weather band.
    Severity:  KSI collisions observed vs expected for the same road types
               (urban/rural, speed limit, daylight/darkness), by weather band.

Tabs and visualisations:
    1. Weather effects     map, frequency by weather, severity by weather
    2. Light and road      rain effect in daylight vs darkness; on urban vs fast roads
    3. When                hour-of-day profile on wet vs dry days; weekly collisions and rainfall
    4. Data check          police-recorded weather vs HadUK rainfall; temperature by road surface
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------

# Project folder = the first folder above this file that contains "data"
# (works whether app.py is in the main folder or inside dashboard/).
PROJECT_DIR = next(p for p in Path(__file__).resolve().parents if (p / "data").is_dir())

COLLISIONS_PATH = PROJECT_DIR / "data" / "merged" / "collisions_weather_2025.csv"
DAILY_PATH = PROJECT_DIR / "data" / "merged" / "daily_gb_weather_collisions_2025.csv"
EXPOSURE_PATH = PROJECT_DIR / "dashboard" / "exposure_counts.npz"

N_DAYS = 365
BLUE = "#2a78d6"      # main series
ORANGE = "#eb6834"    # second series (comparison group)
GREY = "#888780"      # reference lines
LIGHT_GREY = "#b4b2a9"
GRID = "rgba(136, 135, 128, 0.25)"   # light gridlines that work on light and dark themes
MIN_COUNT = 200       # dots based on fewer collisions are greyed out

# Weather bands: identical to notebook 04 (Steps 3.2, 5 and 6).
RAIN_EDGES = [0.2, 1, 5, 10]
RAIN_LABELS = ["Dry (<0.2 mm)", "0.2–1 mm", "1–5 mm", "5–10 mm", "10+ mm"]
DAYS_2025 = pd.date_range("2025-01-01", "2025-12-31", freq="D")
MONTH_WEEKDAY = ((DAYS_2025.month - 1) * 7 + DAYS_2025.dayofweek).to_numpy()   # 84 groups
RAIN3_LABELS = ["Dry", "Light to moderate", "Heavy (10+ mm)"]
TEMP_EDGES = [0, 3, 10, 15, 20]
TEMP_LABELS = ["Below 0 °C", "0–3 °C", "3–10 °C", "10–15 °C", "15–20 °C", "20 °C+"]

st.set_page_config(page_title="Weather and road collisions, GB 2025", layout="wide")


def show_chart(fig):
    """Full-width Plotly chart on both older and newer Streamlit versions."""
    major, minor = (int(p) for p in st.__version__.split(".")[:2])
    if (major, minor) >= (1, 48):
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def is_dark_theme():
    """True when the viewer's Streamlit theme is dark (newer Streamlit versions only)."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def map_colorscale():
    """Sequential blues where more = more prominent on either background.

    On a light page dark blue stands out, so high values are dark; on a dark page
    bright blue stands out, so the scale runs from deep navy (low) to pale blue (high).
    """
    if is_dark_theme():
        return [[0, "#0c2c52"], [0.5, "#2a78d6"], [1, "#d6e8fb"]]
    # Light theme (and fallback): starts at a mid-light blue rather than white,
    # so low-count squares stay visible even if the page is dark.
    return [[0, "#b5d4f4"], [0.5, "#378add"], [1, "#0c447c"]]


def base_layout(fig, title, height=340, y_title=None, legend=False):
    """Shared styling: white background, light gridlines, readable titles."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)), height=height,
        margin=dict(l=10, r=10, t=60, b=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",   # follow the app theme
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


# ---------------------------------------------------------------------------------
# Data loading (cached, so it runs once rather than on every filter change)
# ---------------------------------------------------------------------------------

@st.cache_data
def load_data():
    flag_columns = ["police_reported_rain", "police_reported_snow", "surface_wet",
                    "surface_ice_snow", "coastal_cell", "ksi"]
    collisions = pd.read_csv(
        COLLISIONS_PATH,
        dtype={"collision_index": str, "local_authority_ons_district": str,
               **{c: "boolean" for c in flag_columns}},
        parse_dates=["date", "datetime", "rain_day"],
    )
    daily = pd.read_csv(DAILY_PATH, index_col="day", parse_dates=["day"])
    exposure = np.load(EXPOSURE_PATH)

    # Number each Great Britain cell, and give every collision its cell number.
    cells = pd.DataFrame({"weather_cell_row": exposure["rows"], "weather_cell_col": exposure["cols"]})
    cells["cell"] = np.arange(len(cells))
    collisions = collisions.merge(cells, on=["weather_cell_row", "weather_cell_col"],
                                  how="left", validate="many_to_one")

    # Day index (0 = 1 January) of each collision on the two HadUK-Grid day definitions:
    # rain day (09:00-09:00 UTC, from notebook 03) and temperature day (midnight-midnight UTC).
    day_lookup = pd.Series(np.arange(len(DAYS_2025)), index=DAYS_2025)
    collisions["rain_t"] = day_lookup.reindex(collisions["rain_day"].dt.normalize()).fillna(-1).astype(int).to_numpy()
    utc = (collisions["datetime"].dt.tz_localize("Europe/London", ambiguous=False, nonexistent="shift_forward")
           .dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize())
    collisions["tas_t"] = day_lookup.reindex(utc).fillna(-1).astype(int).to_numpy()

    # Weather band of each collision (-1 where rainfall is blank).
    rain_band = np.digitize(collisions["rainfall_mm"].fillna(-1), RAIN_EDGES)
    collisions["rain_band"] = np.where(collisions["rainfall_mm"].notna(), rain_band, -1)
    collisions["rain3_band"] = np.select(
        [collisions["rain_band"] == 0, collisions["rain_band"] == 4,
         collisions["rain_band"].between(1, 3)], [0, 2, 1], default=-1)
    collisions["temp_band"] = np.digitize(collisions["tas_c"], TEMP_EDGES)

    # Road-type groups used for the severity comparison.
    collisions["ksi_flag"] = collisions["ksi"].astype(bool)
    collisions["darkness"] = collisions["light_conditions_label"].str.startswith("Darkness")
    collisions["speed_group"] = pd.cut(collisions["speed_limit"], bins=[0, 30, 50, 70],
                                       labels=["20–30 mph", "40–50 mph", "60–70 mph"])

    # Each collision's expected chance of KSI = KSI share of its road-type group,
    # calculated once on all collisions, so filters do not change the reference.
    strata = ["urban_or_rural_area_label", "speed_group", "darkness"]
    collisions["stratum_ksi_rate"] = (
        collisions.groupby(strata, observed=True)["ksi_flag"].transform("mean")
    )

    # Days each cell spent in each weather band, by month-weekday group: (cells, bands, 84).
    rain_c = exposure["rain"].astype(float)
    temp_c = exposure["temp"].astype(float)
    # Three rain bands: dry, light to moderate (bands 1-3 combined), heavy.
    rain3_c = np.stack([rain_c[:, 0], rain_c[:, 1:4].sum(axis=1), rain_c[:, 4]], axis=1)
    return collisions, daily, rain_c, rain3_c, temp_c


# ---------------------------------------------------------------------------------
# Analysis functions (notebook 04 methods, applied to the filtered collisions)
# ---------------------------------------------------------------------------------

def frequency_vs_expected(df, band_col, counts, labels, day_col):
    """Collisions observed vs expected for the same cells, months and weekdays.

    Same method as notebook 04, Steps 3.2, 5 and 6. The month-weekday pattern is
    calculated from the selected collisions themselves, so a group such as darkness
    collisions is compared with its own seasonal pattern, not with that of all collisions.
    counts: days each cell spent in each band, by month-weekday group (cells, bands, 84).
    day_col: "rain_t" for rainfall bands, "tas_t" for temperature bands.
    """
    df = df[(df[band_col] >= 0) & df["cell"].notna() & (df[day_col] >= 0)]
    observed = np.bincount(df[band_col], minlength=len(labels)).astype(float)
    if observed.sum() == 0:
        return None

    # Usual collisions per day in each cell, for this selection.
    usual = np.bincount(df["cell"].astype(int), minlength=counts.shape[0]) / N_DAYS

    # How busy each month-weekday group is for this selection, relative to its average day.
    daily_counts = np.bincount(df[day_col], minlength=N_DAYS)
    group_mean = pd.Series(daily_counts).groupby(MONTH_WEEKDAY).mean().reindex(range(84)).fillna(0)
    factor = group_mean.to_numpy() / daily_counts.mean()

    # Expected = sum over cells and month-weekday groups of usual rate x factor x days in band.
    expected = np.einsum("c,cbm->bm", usual, counts) @ factor
    expected *= observed.sum() / expected.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = 100 * (observed / expected - 1)
        ci = 100 * 1.96 * np.sqrt(observed) / expected
    return pd.DataFrame({"band": labels, "collisions": observed.astype(int),
                         "diff_pct": diff, "ci95": ci})


def severity_vs_expected(df, band_col, labels):
    """KSI collisions observed vs expected for the same road types."""
    df = df[df[band_col] >= 0]
    if df.empty:
        return None
    g = df.groupby(band_col).agg(
        collisions=("ksi_flag", "size"),
        ksi=("ksi_flag", "sum"),
        expected=("stratum_ksi_rate", "sum"),
        variance=("stratum_ksi_rate", lambda p: (p * (1 - p)).sum()),
    ).reindex(range(len(labels)))
    g["collisions"] = g["collisions"].fillna(0).astype(int)
    g["band"] = labels
    g["diff_pct"] = 100 * (g["ksi"] / g["expected"] - 1)
    g["ci95"] = 100 * 1.96 * np.sqrt(g["variance"]) / g["expected"]
    return g.reset_index(drop=True)


# ---------------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------------

def dot_interval_chart(table, title, y_title):
    """Dots with 95% intervals around a zero line; thin bands are greyed out."""
    reliable = table["collisions"] >= MIN_COUNT
    labels = [f"{b}<br><span style='font-size:11px'>{n:,}</span>"
              for b, n in zip(table["band"], table["collisions"])]
    fig = go.Figure()
    for mask, colour in [(reliable, BLUE), (~reliable, LIGHT_GREY)]:
        if mask.any():
            fig.add_trace(go.Scatter(
                x=np.array(labels)[mask], y=table.loc[mask, "diff_pct"],
                error_y=dict(type="data", array=table.loc[mask, "ci95"], thickness=1.5, width=5),
                mode="markers", marker=dict(size=11, color=colour),
                hovertemplate="%{x}<br>%{y:+.1f}% vs expected<extra></extra>",
            ))
    fig.add_hline(y=0, line_color=GREY, line_width=1)
    fig.update_xaxes(categoryorder="array", categoryarray=labels)
    fig.update_yaxes(ticksuffix="%")
    return base_layout(fig, title, y_title=y_title)


def paired_dot_chart(tables, title, y_title):
    """Two groups side by side per band (e.g. daylight vs darkness), with intervals."""
    fig = go.Figure()
    offsets = [-0.12, 0.12]
    for (name, table), colour, symbol, offset in zip(tables.items(), [BLUE, ORANGE],
                                                     ["circle", "diamond"], offsets):
        if table is None:
            continue
        x = np.arange(len(table)) + offset
        fig.add_trace(go.Scatter(
            x=x, y=table["diff_pct"], name=name, mode="markers",
            error_y=dict(type="data", array=table["ci95"], thickness=1.5, width=4),
            marker=dict(size=11, color=colour, symbol=symbol),
            customdata=table["collisions"],
            hovertemplate=name + "<br>%{y:+.1f}% vs expected<br>%{customdata:,} collisions<extra></extra>",
        ))
    fig.add_hline(y=0, line_color=GREY, line_width=1)
    fig.update_xaxes(tickvals=list(range(len(RAIN3_LABELS))), ticktext=RAIN3_LABELS)
    fig.update_yaxes(ticksuffix="%")
    return base_layout(fig, title, height=380, y_title=y_title, legend=True)


def collision_map(df, metric, block_km=10):
    """Grid map of collisions or KSI share in square blocks (British National Grid)."""
    size = block_km * 1000
    bx = (df["location_easting_osgr"] // size).astype(int)
    by = (df["location_northing_osgr"] // size).astype(int)
    grid = df.assign(bx=bx, by=by).groupby(["bx", "by"]).agg(
        collisions=("ksi_flag", "size"), ksi=("ksi_flag", "sum"))
    full = pd.MultiIndex.from_product(
        [range(0, 700 // block_km + 1), range(0, 1250 // block_km + 1)], names=["bx", "by"])
    grid = grid.reindex(full)

    if metric == "Collisions":
        # Counts are very skewed (cities vs countryside), so colour on a log scale.
        z = np.log10(grid["collisions"])
        shown = grid["collisions"]
        colour_title, zmin, zmax = "Collisions", 0, max(z.max(), 1)
        hover = "%{customdata:,.0f} collisions<extra></extra>"
        ticks = dict(tickvals=[0, 1, 2, 3], ticktext=["1", "10", "100", "1,000"])
    else:
        # KSI share only where there are enough collisions to be meaningful.
        z = (100 * grid["ksi"] / grid["collisions"]).where(grid["collisions"] >= 20)
        shown = z
        colour_title, zmin, zmax = "KSI share (%)", 10, 50
        hover = "%{customdata:.0f}% KSI<extra></extra>"
        ticks = {}

    z_grid = z.unstack("bx").sort_index()   # rows = northing blocks, columns = easting blocks
    shown_grid = shown.unstack("bx").sort_index()
    fig = go.Figure(go.Heatmap(
        z=z_grid.values, customdata=shown_grid.values,
        x=(z_grid.columns.to_numpy() + 0.5) * block_km,
        y=(z_grid.index.to_numpy() + 0.5) * block_km,
        colorscale=map_colorscale(), zmin=zmin, zmax=zmax, hovertemplate=hover, hoverongaps=False,
        colorbar=dict(title=dict(text=colour_title, side="right"), thickness=12, **ticks),
    ))
    fig.update_layout(
        height=640, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[0, 700]),
        yaxis=dict(visible=False, range=[0, 1230], scaleanchor="x", scaleratio=1),
    )
    return fig


def hour_profile_chart(df):
    """Share of each day's collisions by hour, on wet vs dry days in the collision's cell."""
    fig = go.Figure()
    for name, mask, colour, dash in [
        ("Dry days (<0.2 mm)", df["rainfall_mm"] < 0.2, BLUE, "solid"),
        ("Wet days (1 mm or more)", df["rainfall_mm"] >= 1, ORANGE, "dash"),
    ]:
        counts = df.loc[mask, "hour"].value_counts().reindex(range(24), fill_value=0)
        if counts.sum() == 0:
            continue
        share = 100 * counts / counts.sum()
        fig.add_trace(go.Scatter(
            x=list(range(24)), y=share, name=f"{name}, {counts.sum():,} collisions",
            mode="lines+markers", line=dict(color=colour, width=2, dash=dash),
            marker=dict(size=5), hovertemplate="%{x}:00<br>%{y:.1f}% of collisions<extra></extra>",
        ))
    fig.update_xaxes(title="Hour of day (UK local time)", tickvals=list(range(0, 24, 3)),
                     ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)])
    fig.update_yaxes(ticksuffix="%")
    return base_layout(fig, "When in the day collisions happen: wet vs dry days",
                       height=380, y_title="Share of collisions", legend=True)


def weekly_timeline_chart(df, daily):
    """Weekly collisions (filtered selection) above weekly GB rainfall, on a shared time axis."""
    weekly_collisions = (df.dropna(subset=["rain_day"]).set_index("rain_day")
                         .resample("W-MON", label="left", closed="left").size())
    weekly_rain = daily["gb_mean_rainfall_mm"].resample("W-MON", label="left", closed="left").sum()
    # Drop the partial first and last weeks so they don't look like dips.
    weekly_collisions = weekly_collisions.iloc[1:-1]
    weekly_rain = weekly_rain.reindex(weekly_collisions.index)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.16,
                        row_heights=[0.6, 0.4],
                        subplot_titles=("Collisions per week (selection)",
                                        "Rainfall per week, average across Great Britain (mm)"))
    fig.add_trace(go.Scatter(x=weekly_collisions.index, y=weekly_collisions, mode="lines",
                             line=dict(color=BLUE, width=2),
                             hovertemplate="Week of %{x|%d %b}<br>%{y:,} collisions<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=weekly_rain.index, y=weekly_rain, marker_color="#85B7EB",
                         hovertemplate="Week of %{x|%d %b}<br>%{y:.1f} mm<extra></extra>"),
                  row=2, col=1)
    fig = base_layout(fig, "Through the year: collisions and rainfall by week", height=460)
    fig.update_yaxes(rangemode="tozero")
    return fig


def police_vs_haduk_chart(df):
    """Share of collisions on a HadUK-Grid rain day, by the weather police recorded."""
    groups = {
        "Fine": df["weather_conditions_label"].str.startswith("Fine"),
        "Raining": df["weather_conditions_label"].str.startswith("Raining"),
        "Snowing": df["weather_conditions_label"].str.startswith("Snowing"),
        "Fog or mist": df["weather_conditions_label"] == "Fog or mist",
    }
    rows = []
    for name, mask in groups.items():
        sub = df.loc[mask & df["rainfall_mm"].notna(), "rainfall_mm"]
        if len(sub):
            rows.append((name, 100 * (sub >= 0.2).mean(), sub.median(), len(sub)))
    table = pd.DataFrame(rows, columns=["police", "rain_day_pct", "median_mm", "collisions"])
    fig = go.Figure(go.Bar(
        x=table["rain_day_pct"], y=table["police"], orientation="h", marker_color=BLUE,
        text=[f"{v:.0f}%" for v in table["rain_day_pct"]], textposition="inside",
        insidetextanchor="end", textfont=dict(color="white", size=13),
        customdata=np.column_stack([table["median_mm"], table["collisions"]]),
        hovertemplate="%{y}<br>%{x:.1f}% on a rain day<br>median %{customdata[0]:.2f} mm"
                      "<br>%{customdata[1]:,} collisions<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title="Collisions on a HadUK-Grid rain day (≥ 0.2 mm)")
    fig.update_yaxes(autorange="reversed")
    return base_layout(fig, "Check 1: police-recorded weather vs measured rainfall", height=340)


def surface_temperature_chart(df):
    """Distribution of HadUK-Grid temperature by the road surface police recorded."""
    order = ["Dry", "Wet or damp", "Frost or ice", "Snow"]
    fig = go.Figure()
    for surface in order:
        values = df.loc[df["road_surface_conditions_label"] == surface, "tas_c"]
        if len(values):
            fig.add_trace(go.Box(y=values, name=f"{surface}<br>{len(values):,}", marker_color=BLUE,
                                 boxpoints=False, line=dict(width=1.5)))
    fig.add_hline(y=0, line_color=GREY, line_width=1, line_dash="dot")
    fig.update_yaxes(ticksuffix=" °C")
    return base_layout(fig, "Check 2: measured temperature by police-recorded road surface",
                       height=340, y_title="Daily mean temperature")


# ---------------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------------

collisions, daily, rain_counts, rain3_counts, temp_counts = load_data()

st.title("Weather and road collisions in Great Britain, 2025")
st.caption("Met Office HadUK-Grid daily rainfall and temperature, matched to the 5 km cell and day "
           "of every DfT-recorded collision. Comparisons are against what place, season, weekday "
           "and road type would predict.")

# --- Sidebar filters ------------------------------------------------------------

with st.sidebar:
    st.header("Filters")
    countries = st.multiselect("Country", ["England", "Wales", "Scotland"],
                               default=["England", "Wales", "Scotland"])
    light = st.radio("Light", ["All", "Daylight", "Darkness"], horizontal=True)
    speed = st.selectbox("Road speed limit", ["All roads", "20–30 mph", "40–50 mph", "60–70 mph"])
    severity_filter = st.radio("Severity", ["All collisions", "KSI only"], horizontal=True,
                               help="Applies to the map, hour profile and weekly timeline. "
                                    "The severity charts always use all collisions.")
    st.divider()
    st.subheader("Weather effects tab")
    measure = st.radio("Weather measure", ["Rainfall", "Temperature"], horizontal=True)
    map_metric = st.radio("Map shows", ["Collisions", "KSI share"], horizontal=True)
    map_weather = st.selectbox("Map: days that were",
                               ["All days", "Dry (<0.2 mm)", "Wet (1 mm or more)", "Heavy rain (10+ mm)"])

base = collisions[collisions["country"].isin(countries)]
subset = base
if light != "All":
    subset = subset[subset["darkness"] == (light == "Darkness")]
if speed != "All roads":
    subset = subset[subset["speed_group"] == speed]
counted = subset[subset["ksi_flag"]] if severity_filter == "KSI only" else subset

if subset.empty or counted.empty:
    st.warning("No collisions match these filters. Widen the selection in the sidebar.")
    st.stop()

# --- Headline figures --------------------------------------------------------------

dark_freq = frequency_vs_expected(base[base["darkness"]], "rain_band", rain_counts, RAIN_LABELS, "rain_t")
heavy_dark = dark_freq["diff_pct"].iloc[-1] if dark_freq is not None else np.nan
fast_sev = severity_vs_expected(base[base["speed_group"] == "60–70 mph"], "rain_band", RAIN_LABELS)
heavy_fast = fast_sev["diff_pct"].iloc[-1] if fast_sev is not None else np.nan

k1, k2, k3, k4 = st.columns(4)
k1.metric("Collisions in selection", f"{len(counted):,}")
k2.metric("KSI share", f"{100 * subset['ksi_flag'].mean():.1f}%",
          help="Share of collisions in the selection that killed or seriously injured someone (KSI).")
k3.metric("Heavy rain at night", f"{heavy_dark:+.1f}%",
          help="Change in collisions in darkness on days with 10+ mm in the collision's cell, vs expected "
               "for the same places, months and weekdays (selected countries).")
k4.metric("Heavy rain, fast roads (KSI)", f"{heavy_fast:+.1f}%",
          help="KSI collisions on 60–70 mph roads on 10+ mm days, vs expected for the same "
               "road types (selected countries).")

tab_effects, tab_groups, tab_time, tab_check = st.tabs(
    ["Weather effects", "Light and road type", "When", "Data check"])

# --- Tab 1: map, frequency and severity by weather ---------------------------------------

with tab_effects:
    left, right = st.columns([5, 6])
    with left:
        # KSI share needs all collisions as its denominator, so it ignores the
        # "KSI only" filter; collision counts respect it.
        map_source = subset if map_metric == "KSI share" else counted
        rain = map_source["rainfall_mm"]
        weather_masks = {
            "All days": pd.Series(True, index=map_source.index),
            "Dry (<0.2 mm)": rain < 0.2,
            "Wet (1 mm or more)": rain >= 1,
            "Heavy rain (10+ mm)": rain >= 10,
        }
        map_data = map_source[weather_masks[map_weather]]
        st.subheader(f"Where: {map_metric.lower()} by 10 km square")
        st.caption(f"{map_weather}, {len(map_data):,} collisions. KSI share is shown only "
                   "for squares with at least 20 collisions.")
        show_chart(collision_map(map_data, map_metric))

    with right:
        if measure == "Rainfall":
            band_col, counts, labels, day_col = "rain_band", rain_counts, RAIN_LABELS, "rain_t"
            axis = "rainfall in the collision's 5 km cell"
        else:
            band_col, counts, labels, day_col = "temp_band", temp_counts, TEMP_LABELS, "tas_t"
            axis = "daily mean temperature in the collision's 5 km cell"
        st.subheader(f"How {measure.lower()} changes collisions")
        st.caption(f"By {axis}. Dots show the difference from expected; lines are 95% confidence "
                   f"intervals. Grey dots rest on fewer than {MIN_COUNT} collisions.")
        freq = frequency_vs_expected(subset, band_col, counts, labels, day_col)
        if freq is not None:
            show_chart(dot_interval_chart(
                freq, "How many collisions: vs the same places, months and weekdays",
                "Collisions vs expected"))
        sev = severity_vs_expected(subset, band_col, labels)
        if sev is not None:
            show_chart(dot_interval_chart(
                sev, "How serious: KSI collisions vs the same road types",
                "KSI collisions vs expected"))

# --- Tab 2: rain effect by light and by road type ------------------------------------------

with tab_groups:
    st.subheader("Does rain matter more at night, or on fast roads?")
    st.caption("Uses the country filter only, so the groups can be compared. Rainfall in the "
               "collision's 5 km cell: dry (<0.2 mm), light to moderate (0.2–10 mm), heavy (10+ mm).")
    c1, c2 = st.columns(2)
    with c1:
        light_tables = {
            "Daylight": frequency_vs_expected(base[~base["darkness"]], "rain3_band", rain3_counts, RAIN3_LABELS, "rain_t"),
            "Darkness": frequency_vs_expected(base[base["darkness"]], "rain3_band", rain3_counts, RAIN3_LABELS, "rain_t"),
        }
        show_chart(paired_dot_chart(light_tables, "How many collisions: daylight vs darkness",
                                    "Collisions vs expected"))
    with c2:
        road_tables = {
            "Urban roads (20–30 mph)": severity_vs_expected(base[base["speed_group"] == "20–30 mph"], "rain3_band", RAIN3_LABELS),
            "Fast roads (60–70 mph)": severity_vs_expected(base[base["speed_group"] == "60–70 mph"], "rain3_band", RAIN3_LABELS),
        }
        show_chart(paired_dot_chart(road_tables, "How serious: urban vs fast roads",
                                    "KSI collisions vs expected"))

# --- Tab 3: timing ----------------------------------------------------------------------------

with tab_time:
    st.subheader("When collisions happen")
    st.caption("Uses all sidebar filters. The hour profile compares the share of collisions in each "
               "hour, so wet and dry days can be compared even though there are more dry-day collisions.")
    show_chart(hour_profile_chart(counted))
    show_chart(weekly_timeline_chart(counted, daily))

# --- Tab 4: data checks -----------------------------------------------------------------------

with tab_check:
    st.subheader("Do the weather data agree with what police recorded?")
    st.caption("Independent checks that collisions were matched to the right cell and day. "
               "Agreement is not expected to be perfect: police record conditions at one moment, "
               "HadUK-Grid gives a daily value for a 5 km cell. Uses all sidebar filters.")
    c1, c2 = st.columns(2)
    with c1:
        show_chart(police_vs_haduk_chart(counted))
    with c2:
        show_chart(surface_temperature_chart(counted))

# --- Notes ----------------------------------------------------------------------------

with st.expander("How to read this dashboard, and its limitations"):
    st.markdown(
        """
- **0% means no difference** from what the place, month, weekday (frequency) or road type
  (severity) would predict. A dot whose interval crosses 0% is not a clear difference.
- **Frequency** compares each 5 km cell with its own usual number of collisions on similar days,
  so busy towns are not compared with quiet rural areas.
- **Severity** compares the share of collisions that are fatal or serious (KSI) with the share
  expected for the same urban/rural setting, speed limit and light.
- **Rainfall** is the 24-hour total from 09:00 UTC; **temperature** is the daily mean, the average
  of the day's maximum and minimum. Neither is the weather at the moment of the collision.
- **Traffic volume is not included.** If fewer people travel in bad weather, the risk per mile
  travelled may be higher than these comparisons suggest.
- Northern Ireland is not included: DfT collision data covers Great Britain only.
"""
    )

st.caption("Sources: Met Office HadUK-Grid (daily rainfall and tas, 5 km, 2025); Department for "
           "Transport road safety open data (collisions, 2025); ONS Countries (December 2025) boundaries.")
