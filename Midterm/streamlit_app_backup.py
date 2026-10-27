
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------------------------------------
# DATS 6401 MIDTERM PROJECT
# Buffalo Range Airport Weather & Geospatial Intelligence
# Data source: NASA POWER
# -------------------------------------------------------

st.set_page_config(
    page_title="Buffalo Range Aviation Intelligence",
    page_icon="🇿🇼",
    layout="wide"
)

DATA_PATH = Path(__file__).parent / "buffalo_range_weather.csv"


@st.cache_data
def load_data():
    # Locate the YEAR header after NASA POWER metadata.
    with open(DATA_PATH, "r", encoding="utf-8-sig") as file:
        lines = file.readlines()

    header_index = next(
        i for i, line in enumerate(lines)
        if line.strip().startswith("YEAR,")
    )

    data = pd.read_csv(DATA_PATH, skiprows=header_index)

    data["DATE"] = pd.to_datetime(
        data["YEAR"].astype(str)
        + data["DOY"].astype(str).str.zfill(3),
        format="%Y%j"
    )

    data = data.replace(-999, np.nan)
    return data.sort_values("DATE")


df = load_data()

st.title("🇿🇼 Buffalo Range Aviation Intelligence")
st.caption(
    "DATS 6401 Mid-term | Buffalo Range Airport, Chiredzi, Zimbabwe"
)

st.info(
    "Weather values are NASA POWER gridded estimates for the "
    "airport location, not certified airport observations."
)

# -------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------

st.sidebar.header("Explore the dataset")

date_range = st.sidebar.date_input(
    "Select date range",
    value=(df["DATE"].min().date(), df["DATE"].max().date()),
    min_value=df["DATE"].min().date(),
    max_value=df["DATE"].max().date()
)

if len(date_range) != 2:
    st.warning("Please select both start and end dates.")
    st.stop()

start_date, end_date = date_range

filtered = df[
    (df["DATE"].dt.date >= start_date)
    & (df["DATE"].dt.date <= end_date)
].copy()

if filtered.empty:
    st.warning("No data available for the selected period.")
    st.stop()

# -------------------------------------------------------
# MAIN TABS
# -------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Airport Overview",
    "📈 Weather Time-Series",
    "🌬️ Wind Analysis",
    "📖 Interpretation"
])

with tab1:
    st.subheader("Buffalo Range Airport — Overview")

    st.markdown(
        "**Location:** Chiredzi, Zimbabwe  \n"
        "**ICAO:** FVCZ  \n"
        "**Data source:** NASA POWER"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Average temperature",
        f"{filtered['T2M'].mean():.1f} °C"
    )
    col2.metric(
        "Average wind speed",
        f"{filtered['WS10M'].mean():.2f} m/s"
    )
    col3.metric(
        "Total rainfall",
        f"{filtered['PRECTOTCORR'].sum():,.1f} mm"
    )
    col4.metric(
        "Days analyzed",
        f"{len(filtered):,}"
    )

    st.markdown(
        "Explore historical weather variability around "
        "Buffalo Range Airport and its relevance to "
        "aviation infrastructure research."
    )

with tab2:
    st.subheader("Historical Weather Patterns")

    variable_labels = {
        "Temperature (°C)": "T2M",
        "Maximum temperature (°C)": "T2M_MAX",
        "Wind speed at 10 m (m/s)": "WS10M",
        "Maximum wind speed (m/s)": "WS10M_MAX",
        "Rainfall (mm/day)": "PRECTOTCORR",
        "Relative humidity (%)": "RH2M",
        "Surface pressure (kPa)": "PS",
    }

    selected_label = st.selectbox(
        "Select weather variable",
        list(variable_labels.keys())
    )

    selected_column = variable_labels[selected_label]

    fig = px.line(
        filtered,
        x="DATE",
        y=selected_column,
        title=f"Buffalo Range: {selected_label}",
        labels={
            "DATE": "Date",
            selected_column: selected_label
        }
    )

    fig.update_layout(
        height=480,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Interactive Wind Rose")

    st.markdown(
        "The wind rose summarizes **daily mean wind direction "
        "and wind speed at 10 metres** using NASA POWER data."
    )

    wind = filtered[["WD10M", "WS10M"]].dropna().copy()

    # Wind directions are circular: 0° and 360° both mean North.
    directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW"
    ]

    # Assign each daily observation to one of 16 directions.
    sector_index = (
        np.floor(((wind["WD10M"] % 360) + 11.25) / 22.5)
        .astype(int) % 16
    )

    wind["Direction"] = [
        directions[index] for index in sector_index
    ]

    # Categorize daily mean wind speeds.
    wind["Speed category"] = pd.cut(
        wind["WS10M"],
        bins=[-np.inf, 2, 4, 6, 8, np.inf],
        labels=[
            "0–2 m/s",
            "2–4 m/s",
            "4–6 m/s",
            "6–8 m/s",
            ">8 m/s"
        ]
    )

    # Count observations per direction and speed category.
    wind_counts = (
        wind.groupby(
            ["Direction", "Speed category"],
            observed=False
        )
        .size()
        .reset_index(name="Days")
    )

    wind_counts["Frequency (%)"] = (
        100 * wind_counts["Days"] / len(wind)
    )

    fig_wind = px.bar_polar(
        wind_counts,
        r="Frequency (%)",
        theta="Direction",
        color="Speed category",
        category_orders={
            "Direction": directions,
            "Speed category": [
                "0–2 m/s",
                "2–4 m/s",
                "4–6 m/s",
                "6–8 m/s",
                ">8 m/s"
            ]
        },
        title="Buffalo Range — Daily Wind Direction and Speed",
        labels={
            "Frequency (%)": "Frequency (% of days)"
        }
    )

    fig_wind.update_layout(
        height=650,
        polar=dict(
            angularaxis=dict(
                direction="clockwise",
                rotation=90
            )
        ),
        legend_title_text="Daily mean wind speed"
    )

    st.plotly_chart(fig_wind, use_container_width=True)

    st.caption(
        "Each sector shows the percentage of analyzed days "
        "with wind originating from that direction. "
        "Daily mean winds cannot capture short-lived gusts "
        "or runway-level crosswind conditions."
    )

with tab4:
    st.subheader("Interpretation and Methodology")

    st.markdown(
        """
        ### Research objective
        Examine historical meteorological variability at
        Buffalo Range Airport, Chiredzi, Zimbabwe.

        ### Dataset
        NASA POWER daily meteorological data for the
        selected airport coordinates, 2015–2025.

        ### Visualization methods
        - Interactive time-series for temporal patterns
        - Wind rose for directional and speed distributions
        - Date filtering for historical comparisons

        ### Limitations
        NASA POWER provides gridded weather estimates,
        not certified airport observations. Daily wind
        averages do not establish operational runway safety.
        """
    )

