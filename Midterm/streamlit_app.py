
from pathlib import Path
import math
import html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

# ============================================================
# BUFFALO RANGE AVIATION INTELLIGENCE
# Visual Command Center | DATS 6401
# ============================================================

st.set_page_config(
    page_title="Buffalo Range Aviation Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "buffalo_range_weather.csv"
HERO_PATH = BASE / "airport_hero.jpg"

LAT = -21.0167
LON = 31.5833

BLUE = "#034694"
LIGHT_BLUE = "#2D85CC"
GREEN = "#E8F5E9"
NAVY = "#082B59"

# ============================================================
# CSS — CHELSEA BLUE + LIGHT GREEN
# ============================================================

st.markdown("""
<style>
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #E8F5E9 !important;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 1650px !important;
}

[data-testid="stSidebar"] {
    background: #D8EEE2 !important;
    border-right: 1px solid #BEDCCB;
}

[data-testid="stSidebar"] *,
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] p,
[data-testid="stMain"] label {
    color: #034694;
}

[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #C8E2D2;
    border-top: 4px solid #034694;
    border-radius: 11px;
    padding: 10px 12px;
    box-shadow: 0 2px 8px rgba(3,70,148,.07);
}

[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"] *,
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #034694 !important;
}

[data-testid="stMetricValue"] {
    font-size: clamp(1.15rem, 1.7vw, 1.75rem) !important;
}

button[data-baseweb="tab"],
button[data-baseweb="tab"] * {
    color: #034694 !important;
    font-weight: 650;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: #E3EEFF !important;
    border-radius: 8px 8px 0 0;
}

[data-testid="stPlotlyChart"],
[data-testid="stDeckGlJsonChart"] {
    background: white;
    border-radius: 12px;
    overflow: hidden;
}

.hero {
    min-height: 170px;
    background:
        linear-gradient(
            90deg,
            rgba(3,35,86,.95) 0%,
            rgba(3,70,148,.88) 42%,
            rgba(3,70,148,.15) 100%
        );
    border-radius: 15px;
    padding: 24px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 15px rgba(3,70,148,.15);
}

.hero-title {
    color: white !important;
    font-size: clamp(1.55rem, 2.7vw, 2.65rem);
    font-weight: 900;
    line-height: 1.06;
    letter-spacing: -.5px;
}

.hero-sub {
    color: #D9EEFF !important;
    font-size: .92rem;
    margin-top: 10px;
}

.hero-credit {
    background: rgba(3,35,86,.75);
    border-radius: 9px;
    color: white !important;
    text-align: right;
    padding: 10px 13px;
    font-size: .79rem;
    line-height: 1.6;
    min-width: 180px;
}

.hero-credit strong {
    color: white !important;
}

.section-label {
    font-weight: 800;
    color: #034694;
    font-size: 1.1rem;
    margin: 8px 0;
}

.insight {
    background: #EAF3FF;
    border-left: 4px solid #034694;
    padding: 10px 13px;
    border-radius: 8px;
    color: #034694;
    font-size: .88rem;
}

.note {
    color: #45678A;
    font-size: .78rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD NASA POWER DATA
# ============================================================

@st.cache_data
def load_data():
    with open(DATA_PATH, encoding="utf-8-sig") as file:
        lines = file.readlines()

    header_index = next(
        i for i, line in enumerate(lines)
        if line.strip().startswith("YEAR,")
    )

    data = pd.read_csv(
        DATA_PATH,
        skiprows=header_index,
    )

    data["DATE"] = pd.to_datetime(
        data["YEAR"].astype(str)
        + data["DOY"].astype(str).str.zfill(3),
        format="%Y%j",
    )

    return (
        data.replace(-999, np.nan)
        .sort_values("DATE")
        .reset_index(drop=True)
    )


try:
    df = load_data()
except (OSError, ValueError, StopIteration) as error:
    st.error(f"Cannot read NASA POWER dataset: {error}")
    st.stop()


# ============================================================
# GEOGRAPHIC HELPERS
# ============================================================

def destination(bearing, distance_km):
    angle = math.radians(bearing)
    radius = 6371.0

    lat1 = math.radians(LAT)
    lon1 = math.radians(LON)
    delta = distance_km / radius

    lat2 = math.asin(
        math.sin(lat1) * math.cos(delta)
        + math.cos(lat1) * math.sin(delta)
        * math.cos(angle)
    )

    lon2 = lon1 + math.atan2(
        math.sin(angle) * math.sin(delta)
        * math.cos(lat1),
        math.cos(delta)
        - math.sin(lat1) * math.sin(lat2),
    )

    return [math.degrees(lon2), math.degrees(lat2)]


DIRECTIONS = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
]

SPEED_NAMES = [
    "0–2", "2–4", "4–6", "6–8", ">8"
]

SPEED_COLORS = [
    [177, 222, 245, 235],
    [91, 166, 221, 235],
    [3, 70, 148, 235],
    [245, 163, 68, 240],
    [209, 63, 71, 245],
]

HEX_COLORS = [
    "#B1DEF5",
    "#5BA6DD",
    "#034694",
    "#F5A344",
    "#D13F47",
]


def wind_frequencies(data):
    wind = data[["WD10M", "WS10M"]].dropna().copy()

    if wind.empty:
        return pd.DataFrame()

    wind["sector"] = (
        np.floor(
            ((wind["WD10M"] % 360) + 11.25) / 22.5
        ).astype(int) % 16
    )

    wind["speed_class"] = pd.cut(
        wind["WS10M"],
        bins=[-np.inf, 2, 4, 6, 8, np.inf],
        labels=False,
    )

    result = (
        wind.groupby(
            ["sector", "speed_class"]
        )
        .size()
        .reset_index(name="count")
    )

    result["pct"] = (
        100 * result["count"] / len(wind)
    )

    return result


def wind_blade(bearing, inner, outer):
    angles = np.linspace(
        bearing - 10.5,
        bearing + 10.5,
        9,
    )

    return (
        [destination(a, outer) for a in angles]
        + [destination(a, inner) for a in angles[::-1]]
    )


def geographic_wind_layers(data, radius):
    frequency = wind_frequencies(data)

    if frequency.empty:
        return []

    maximum = max(
        float(
            frequency.groupby("sector")["pct"]
            .sum().max()
        ),
        0.01,
    )

    polygons = []

    for sector in range(16):
        cumulative = 0

        for speed_class in range(5):
            rows = frequency[
                (frequency["sector"] == sector)
                & (
                    frequency["speed_class"]
                    == speed_class
                )
            ]

            if rows.empty:
                continue

            value = float(rows["pct"].iloc[0])

            if value <= 0:
                continue

            inner = radius * cumulative / maximum
            cumulative += value
            outer = radius * cumulative / maximum

            polygons.append({
                "polygon": wind_blade(
                    sector * 22.5,
                    inner,
                    outer,
                ),
                "color": SPEED_COLORS[speed_class],
                "name": (
                    f"{DIRECTIONS[sector]} | "
                    f"{SPEED_NAMES[speed_class]} m/s | "
                    f"{value:.1f}%"
                ),
            })

    compass_labels = [
        {
            "position": destination(
                sector * 22.5,
                radius * 1.16,
            ),
            "label": DIRECTIONS[sector],
        }
        for sector in range(0, 16, 2)
    ]

    return [
        pdk.Layer(
            "PolygonLayer",
            data=polygons,
            get_polygon="polygon",
            get_fill_color="color",
            get_line_color=[255, 255, 255, 230],
            stroked=True,
            line_width_min_pixels=1,
            pickable=True,
        ),
        pdk.Layer(
            "TextLayer",
            data=compass_labels,
            get_position="position",
            get_text="label",
            get_size=16,
            get_color=[3, 70, 148, 255],
            get_text_anchor="'middle'",
        ),
    ]


def style_chart(fig, height=390):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=BLUE),
        title_font=dict(color=BLUE, size=17),
        height=height,
        margin=dict(l=20, r=15, t=50, b=20),
    )
    return fig


def wind_chart(data, height=430):
    counts = wind_frequencies(data)

    if counts.empty:
        return None

    counts["Direction"] = counts["sector"].map(
        lambda x: DIRECTIONS[int(x)]
    )

    counts["Speed"] = counts["speed_class"].map(
        lambda x: SPEED_NAMES[int(x)]
    )

    fig = px.bar_polar(
        counts,
        r="pct",
        theta="Direction",
        color="Speed",
        category_orders={
            "Direction": DIRECTIONS,
            "Speed": SPEED_NAMES,
        },
        color_discrete_sequence=HEX_COLORS,
        title="Wind Direction and Speed",
    )

    style_chart(fig, height)

    fig.update_layout(
        polar=dict(
            bgcolor="#FFFFFF",
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
            ),
        ),
        legend_title_text="Wind speed (m/s)",
    )

    return fig


def climate_heatmap(data):
    variables = {
        "T2M": "Temp",
        "WS10M": "Wind",
        "PRECTOTCORR": "Rain",
        "RH2M": "Humidity",
        "PS": "Pressure",
    }

    corr = data[list(variables)].corr()

    labels = list(variables.values())

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[
                [0, "#E57A50"],
                [0.5, "#FFFFFF"],
                [1, "#034694"],
            ],
            text=np.round(corr.values, 2),
            texttemplate="%{text:.2f}",
            textfont=dict(size=13),
            colorbar=dict(title="r"),
        )
    )

    return style_chart(fig, 430)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## ✈️ Intelligence Controls")

period = st.sidebar.date_input(
    "Analysis period",
    value=(
        df["DATE"].min().date(),
        df["DATE"].max().date(),
    ),
    min_value=df["DATE"].min().date(),
    max_value=df["DATE"].max().date(),
)

if len(period) != 2:
    st.stop()

data = df[
    (df["DATE"].dt.date >= period[0])
    & (df["DATE"].dt.date <= period[1])
].copy()

if data.empty:
    st.warning("No data in selected period.")
    st.stop()

st.sidebar.markdown("### Map display options")

show_planes = st.sidebar.checkbox(
    "Show aircraft symbols",
    value=True,
)

show_rose = st.sidebar.checkbox(
    "Show wind rose",
    value=True,
)

show_marker = st.sidebar.checkbox(
    "Show airport marker",
    value=True,
)

radius = st.sidebar.slider(
    "Wind rose radius (km)",
    3, 20, 8,
)

climate_layer = st.sidebar.selectbox(
    "Climate map layer",
    [
        "Wind Rose",
        "Temperature",
        "Rainfall",
        "Humidity",
        "NDVI (future)",
        "Terrain (future)",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "DATS 6401 | NASA POWER Daily Dataset"
)


# ============================================================
# HERO BANNER
# ============================================================

# If airport_hero.jpg exists, use it as the visual banner.
# Otherwise use the professional blue gradient.

if HERO_PATH.exists():
    import base64

    with open(HERO_PATH, "rb") as image_file:
        encoded = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    hero_background = (
        "background-image:"
        "linear-gradient(90deg,"
        "rgba(3,35,86,.95) 0%,"
        "rgba(3,70,148,.78) 40%,"
        "rgba(3,70,148,.08) 100%),"
        f"url('data:image/jpeg;base64,{encoded}');"
        "background-size:cover;"
        "background-position:center;"
    )
else:
    hero_background = ""

st.markdown(
    f"""
    <div class="hero" style="{hero_background}">
        <div>
            <div class="hero-title">
                ✈ BUFFALO RANGE<br>
                AVIATION INTELLIGENCE
            </div>
            <div class="hero-sub">
                Geospatial Weather & Environmental Analytics<br>
                Chiredzi, Zimbabwe | FVCZ
            </div>
        </div>
        <div class="hero-credit">
            DEVELOPED BY<br>
            <strong>S. SIMBANEGAVI</strong><br>
            Data Scientist<br>
            Washington, DC, USA
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ============================================================
# SIX KPI CARDS
# ============================================================

metrics = st.columns(6)

metrics[0].metric(
    "🌡️ Temperature",
    f"{data['T2M'].mean():.1f} °C",
)

metrics[1].metric(
    "💨 Wind speed",
    f"{data['WS10M'].mean():.2f} m/s",
)

metrics[2].metric(
    "🌧️ Rainfall",
    f"{data['PRECTOTCORR'].sum():,.0f} mm",
)

metrics[3].metric(
    "💧 Humidity",
    f"{data['RH2M'].mean():.1f}%",
)

metrics[4].metric(
    "🧭 Pressure",
    f"{data['PS'].mean():.1f} kPa",
)

metrics[5].metric(
    "📅 Days",
    f"{len(data):,}",
)


# ============================================================
# NAVIGATION
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Geospatial Intelligence",
    "📈 Weather Trends",
    "💨 Wind Rose Analysis",
    "🔷 Climate Relationships",
    "📖 Research",
])


# ============================================================
# TAB 1 — TWO-COLUMN COMMAND CENTER
# ============================================================

with tab1:

    left, right = st.columns(
        [1.15, 0.85],
        gap="medium",
    )

    with left:
        st.markdown(
            '<div class="section-label">'
            '🗺️ Interactive Airport Climate Map'
            '</div>',
            unsafe_allow_html=True,
        )

        layers = []

        if show_rose and climate_layer == "Wind Rose":
            layers.extend(
                geographic_wind_layers(data, radius)
            )

        if climate_layer in [
            "Temperature", "Rainfall", "Humidity"
        ]:
            settings = {
                "Temperature": (
                    "T2M", "mean", "°C",
                    [244, 159, 67, 225],
                ),
                "Rainfall": (
                    "PRECTOTCORR", "sum", "mm",
                    [65, 143, 220, 225],
                ),
                "Humidity": (
                    "RH2M", "mean", "%",
                    [62, 183, 151, 225],
                ),
            }

            col, agg, unit, color = settings[
                climate_layer
            ]

            value = (
                data[col].sum()
                if agg == "sum"
                else data[col].mean()
            )

            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=[{
                        "position": [LON, LAT],
                        "radius": 650,
                        "name": (
                            f"{climate_layer}: "
                            f"{value:.1f} {unit}"
                        ),
                    }],
                    get_position="position",
                    get_radius="radius",
                    get_fill_color=color,
                    pickable=True,
                )
            )

        if climate_layer in [
            "NDVI (future)",
            "Terrain (future)",
        ]:
            st.info(
                "Satellite or elevation data required."
            )

        if show_planes:
            planes = [
                {
                    "position": destination(55, 3),
                    "label": "✈",
                },
                {
                    "position": destination(235, 3),
                    "label": "✈",
                },
            ]

            layers.append(
                pdk.Layer(
                    "TextLayer",
                    data=planes,
                    get_position="position",
                    get_text="label",
                    get_size=26,
                    get_color=[3, 70, 148, 255],
                )
            )

        if show_marker:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=[{
                        "position": [LON, LAT],
                        "radius": 130,
                        "name": (
                            "Buffalo Range Airport"
                        ),
                    }],
                    get_position="position",
                    get_radius="radius",
                    get_fill_color=[
                        255, 204, 0, 255
                    ],
                    pickable=True,
                )
            )

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(
                latitude=LAT,
                longitude=LON,
                zoom=10,
                pitch=0,
            ),
            map_style=(
                "https://basemaps.cartocdn.com/gl/"
                "positron-gl-style/style.json"
            ),
            tooltip={
                "html": "<b>{name}</b>",
                "style": {
                    "backgroundColor": BLUE,
                    "color": "white",
                },
            },
        )

        st.pydeck_chart(
            deck,
            use_container_width=True,
            height=435,
            key="command_center_map",
        )

        st.markdown(
            "<div class='note'>"
            "Wind speed (m/s): "
            "🔹 0–2 &nbsp; 🔵 2–4 &nbsp; "
            "💙 4–6 &nbsp; 🟠 6–8 &nbsp; 🔴 >8"
            "</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div class="section-label">'
            '💨 Wind Direction and Speed'
            '</div>',
            unsafe_allow_html=True,
        )

        wind_fig = wind_chart(data, 435)

        if wind_fig is not None:
            st.plotly_chart(
                wind_fig,
                use_container_width=True,
                key="command_center_wind",
            )

        counts = wind_frequencies(data)

        if not counts.empty:
            totals = counts.groupby(
                "sector"
            )["pct"].sum()

            top = int(totals.idxmax())

            st.markdown(
                f"""
                <div class="insight">
                💡 <strong>Key Wind Insight</strong><br>
                Most frequent wind-from direction:
                <strong>{DIRECTIONS[top]}</strong>
                ({totals.max():.1f}% of days).
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# TAB 2 — WEATHER TRENDS
# ============================================================

with tab2:
    st.subheader("Historical Weather Trends")

    choices = {
        "Temperature": "T2M",
        "Maximum Temperature": "T2M_MAX",
        "Wind Speed": "WS10M",
        "Rainfall": "PRECTOTCORR",
        "Humidity": "RH2M",
        "Pressure": "PS",
    }

    chosen = st.selectbox(
        "Select variable",
        list(choices),
    )

    field = choices[chosen]

    fig = px.line(
        data,
        x="DATE",
        y=field,
        title=f"Historical {chosen}",
        color_discrete_sequence=[BLUE],
    )

    st.plotly_chart(
        style_chart(fig, 470),
        use_container_width=True,
        key="weather_trend_main",
    )


# ============================================================
# TAB 3 — WIND ANALYSIS
# ============================================================

with tab3:
    st.subheader("Wind Rose Analysis")

    fig = wind_chart(data, 540)

    if fig is not None:
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="wind_rose_full",
        )

    st.caption(
        "Frequency of daily mean wind-from directions "
        "and wind-speed classes."
    )


# ============================================================
# TAB 4 — CLIMATE RELATIONSHIPS
# ============================================================

with tab4:
    st.subheader("Climate Relationship Heatmap")

    fig = climate_heatmap(data)

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="climate_heatmap_main",
    )

    fields = [
        "T2M",
        "WS10M",
        "PRECTOTCORR",
        "RH2M",
        "PS",
    ]

    corr = data[fields].corr()

    pairs = []

    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            value = corr.iloc[i, j]

            if pd.notna(value):
                pairs.append((
                    abs(value),
                    value,
                    fields[i],
                    fields[j],
                ))

    if pairs:
        _, r, a, b = max(pairs)

        st.markdown(
            f"""
            <div class="insight">
            💡 Strongest linear relationship:
            <strong>{html.escape(a)} ↔
            {html.escape(b)}</strong>
            | Pearson r = <strong>{r:+.2f}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Blue: positive correlation. "
        "Orange: negative correlation. "
        "Correlation does not imply causation."
    )


# ============================================================
# TAB 5 — RESEARCH
# ============================================================

with tab5:
    st.subheader("Research and Interpretation")

    st.markdown("""
**Research question**

How do historical weather conditions vary at
Buffalo Range Airport's reference location?

**Data**

NASA POWER daily meteorological estimates.

**Visualization methods**

- Interactive geographic wind rose
- Daily weather time series
- Directional wind-frequency analysis
- Correlation heatmap

**Limitations**

Weather estimates are from a single reference
location and are not certified aviation observations.
Aircraft icons are illustrative, not live traffic.

**Future final-project work**

Satellite NDVI, terrain layers, advanced spatial
analysis and aircraft animation.
    """)

st.caption(
    "© 2026 S. Simbanegavi | "
    "Data Scientist | Washington, DC, USA | "
    "Research and educational use only"
)



