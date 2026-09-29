import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Lumpy Skin Disease in Cattle in Zimbabwe",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

@st.cache_data
def load_data():
    data_path = Path(__file__).parent / "LSD 1995 -2014.csv"
    df = pd.read_csv(data_path)

    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
    }

    df["MONTH"] = df["MONTH"].astype(str).str.upper().str.strip()
    df["MONTH_NUM"] = df["MONTH"].map(month_map)

    df["DATE"] = pd.to_datetime(
        dict(
            year=df["YEAR"],
            month=df["MONTH_NUM"],
            day=1
        ),
        errors="coerce"
    )

    df["CASES"] = pd.to_numeric(df["CASES"], errors="coerce")

    return df


df = load_data()


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Time Series Controls")

resolution = st.sidebar.selectbox(
    "Select temporal resolution:",
    ["Monthly", "Quarterly", "Yearly"]
)

year_range = st.sidebar.slider(
    "Select analysis period:",
    min_value=2011,
    max_value=2013,
    value=(2011, 2013),
    step=1
)

start_year, end_year = year_range

filtered_df = df[
    (df["YEAR"] >= start_year) &
    (df["YEAR"] <= end_year)
].copy()


# ============================================================
# AGGREGATE MONTHLY DATA
# ============================================================

monthly = (
    filtered_df
    .groupby("DATE", as_index=True)["CASES"]
    .sum(min_count=1)
    .to_frame("Total_Cases")
    .sort_index()
)

if not monthly.empty:
    full_dates = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-01",
        freq="MS"
    )

    monthly = monthly.reindex(full_dates)
    monthly.index.name = "DATE"

monthly["Rolling_12M"] = (
    monthly["Total_Cases"]
    .rolling(window=12, min_periods=4)
    .mean()
)

monthly["Rolling_STD"] = (
    monthly["Total_Cases"]
    .rolling(window=12, min_periods=4)
    .std()
)


# ============================================================
# TEMPORAL RESOLUTION
# ============================================================

if resolution == "Monthly":
    display_data = monthly["Total_Cases"].copy()
    resolution_message = (
        "Monthly resolution preserves short-term variation "
        "and makes seasonal patterns visible."
    )

elif resolution == "Quarterly":
    display_data = monthly["Total_Cases"].resample("QE").sum(min_count=1)
    resolution_message = (
        "Quarterly aggregation smooths monthly fluctuations "
        "while retaining broad seasonal changes."
    )

else:
    display_data = monthly["Total_Cases"].resample("YE").sum(min_count=1)
    resolution_message = (
        "Yearly aggregation emphasizes long-term differences "
        "but hides within-year seasonal variation."
    )


# ============================================================
# HEADER
# ============================================================

st.title("Lumpy Skin Disease in Cattle in Zimbabwe")

st.subheader(
    f"Interactive Time Series and Seasonal Analysis, "
    f"{start_year}–{end_year}"
)

st.write(
    "This interactive application explores reported Lumpy Skin Disease "
    "(LSD) cases in cattle in Zimbabwe. Users can change the temporal "
    "resolution and analysis period to examine temporal trends, seasonal "
    "patterns, and variability in reported cases."
)

st.success("Dataset loaded successfully.")


# ============================================================
# SUMMARY METRICS USING COLUMNS
# ============================================================

total_cases = monthly["Total_Cases"].sum()

if monthly["Total_Cases"].notna().any():
    peak_date = monthly["Total_Cases"].idxmax()
    peak_month = peak_date.strftime("%B %Y")
    peak_cases = monthly["Total_Cases"].max()
else:
    peak_month = "N/A"
    peak_cases = 0

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Reported Cases",
    f"{total_cases:,.0f}"
)

col2.metric(
    "Peak Month",
    peak_month
)

col3.metric(
    "Peak Monthly Cases",
    f"{peak_cases:,.0f}"
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Time Series", "Seasonality", "Data"]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.header("Overview")

    st.info(resolution_message)

    st.write(
        f"The current view covers **{start_year}–{end_year}** "
        f"using **{resolution.lower()}** temporal resolution."
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        display_data.index,
        display_data.values,
        marker="o"
    )

    ax.set_title(
        f"{resolution} Reported Lumpy Skin Disease Cases"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Reported Cases")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    if resolution == "Monthly":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%b\n%Y")
        )

    elif resolution == "Quarterly":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%b\n%Y")
        )

    else:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y")
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# TAB 2 — TIME SERIES
# ============================================================

with tab2:

    st.header("Time Series Trend")

    st.write(
        "The monthly series shows short-term variation in reported LSD "
        "cases. The 12-month rolling mean summarizes the broader temporal "
        "pattern."
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        monthly.index,
        monthly["Total_Cases"],
        marker="o",
        alpha=0.6,
        label="Monthly Cases"
    )

    ax.plot(
        monthly.index,
        monthly["Rolling_12M"],
        linewidth=3,
        label="12-Month Rolling Mean"
    )

    ax.set_title(
        f"Monthly LSD Cases with 12-Month Trend, "
        f"{start_year}–{end_year}"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Total Reported Cases")
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=2)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b\n%Y")
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Trend and Uncertainty")

    lower = (
        monthly["Rolling_12M"] -
        2 * monthly["Rolling_STD"]
    ).clip(lower=0)

    upper = (
        monthly["Rolling_12M"] +
        2 * monthly["Rolling_STD"]
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        monthly.index,
        monthly["Total_Cases"],
        marker="o",
        alpha=0.5,
        label="Monthly Cases"
    )

    ax.plot(
        monthly.index,
        monthly["Rolling_12M"],
        linewidth=3,
        label="12-Month Rolling Mean"
    )

    ax.fill_between(
        monthly.index,
        lower,
        upper,
        alpha=0.2,
        label="±2 Rolling Standard Deviations"
    )

    ax.set_title(
        "LSD Trend and Local Variability"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Total Reported Cases")
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=2)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b\n%Y")
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "The shaded region represents ±2 rolling standard deviations "
        "around the 12-month rolling mean. It communicates local "
        "variability and is not a formal confidence interval."
    )


# ============================================================
# TAB 3 — SEASONALITY
# ============================================================

with tab3:

    st.header("Seasonal Pattern of Lumpy Skin Disease")

    seasonal = monthly.copy()
    seasonal["YEAR"] = seasonal.index.year
    seasonal["MONTH_NUM"] = seasonal.index.month

    fig, ax = plt.subplots(figsize=(12, 5))

    for year in range(start_year, end_year + 1):

        year_data = seasonal[
            seasonal["YEAR"] == year
        ]

        ax.plot(
            year_data["MONTH_NUM"],
            year_data["Total_Cases"],
            marker="o",
            label=str(year)
        )

    ax.set_xticks(range(1, 13))

    ax.set_xticklabels(
        [
            "Jan", "Feb", "Mar", "Apr",
            "May", "Jun", "Jul", "Aug",
            "Sep", "Oct", "Nov", "Dec"
        ]
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Total Reported Cases")
    ax.set_title(
        f"Seasonal Pattern of Monthly LSD Cases, "
        f"{start_year}–{end_year}"
    )
    ax.set_ylim(bottom=0)
    ax.legend(title="Year")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Average Monthly Pattern")

    seasonal_average = (
        seasonal
        .groupby("MONTH_NUM")["Total_Cases"]
        .mean()
        .reindex(range(1, 13))
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(
        range(1, 13),
        seasonal_average.values
    )

    ax.set_xticks(range(1, 13))

    ax.set_xticklabels(
        [
            "Jan", "Feb", "Mar", "Apr",
            "May", "Jun", "Jul", "Aug",
            "Sep", "Oct", "Nov", "Dec"
        ]
    )

    ax.set_title(
        f"Average Monthly LSD Cases, "
        f"{start_year}–{end_year}"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Average Reported Cases")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if seasonal_average.notna().any():

        highest_month_number = seasonal_average.idxmax()

        month_names = {
            1: "January",
            2: "February",
            3: "March",
            4: "April",
            5: "May",
            6: "June",
            7: "July",
            8: "August",
            9: "September",
            10: "October",
            11: "November",
            12: "December"
        }

        st.write(
            f"The highest average reported LSD cases in the selected "
            f"period occur in **{month_names[highest_month_number]}**. "
            "The monthly profile helps identify recurring within-year "
            "variation in reported disease cases."
        )


# ============================================================
# TAB 4 — DATA
# ============================================================

with tab4:

    st.header("Filtered Data")

    st.write(
        "The table below contains observations from the selected "
        "analysis period."
    )

    preview_columns = [
        "YEAR",
        "MONTH",
        "PROVINCE",
        "DISTRICT",
        "TYPE OF LOCALITY",
        "CASES",
        "DEATHS"
    ]

    available_columns = [
        column for column in preview_columns
        if column in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[available_columns],
        use_container_width=True
    )

    csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name=(
            f"zimbabwe_lsd_{start_year}_{end_year}.csv"
        ),
        mime="text/csv"
    )

    st.subheader("Temporal Honesty and Data Limitations")

    st.write(
        "December 2011 has no observation in the source dataset. "
        "When that month falls within the selected period, it is retained "
        "as missing rather than being assigned a value of zero because an "
        "absent record does not necessarily mean that zero disease cases "
        "occurred."
    )

    st.write(
        "Changing temporal resolution also changes the visual "
        "interpretation of the data. Monthly resolution reveals "
        "short-term and seasonal variation, while quarterly and yearly "
        "aggregation smooth those variations."
    )

    st.write(
        "The charts use a zero baseline for reported case counts so that "
        "differences are not visually exaggerated."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DATS 6401 — Visualization of Complex Data | "
    "Interactive Streamlit Application"
    " By SIMBANEGAVI SIMBARASHE G31501140"
)