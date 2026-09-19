import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Lumpy Skin Disease in Cattle in Zimbabwe",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# TITLE AND INTRODUCTION
# =========================================================

st.title("Lumpy Skin Disease in Cattle in Zimbabwe")
st.subheader("Time Series and Seasonal Analysis, 2011–2013")

st.write(
    "This interactive application explores monthly reported Lumpy Skin "
    "Disease (LSD) cases in cattle in Zimbabwe from 2011 to 2013. "
    "The analysis examines temporal trends, seasonal patterns, temporal "
    "resolution, and variability in reported cases."
)


# =========================================================
# LOAD AND PREPARE DATA
# =========================================================

@st.cache_data
def load_data():

    data_path = Path(__file__).parent / "LSD 1995 -2014.csv"

    df = pd.read_csv(data_path)

    # Convert month abbreviations to numeric values
    month_map = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12
    }

    df["MONTH_NUM"] = (
        df["MONTH"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(month_map)
    )

    # Create a proper datetime variable
    df["DATE"] = pd.to_datetime(
        dict(
            year=df["YEAR"],
            month=df["MONTH_NUM"],
            day=1
        )
    )

    # Focus analysis on 2011–2013
    df = df[
        (df["YEAR"] >= 2011) &
        (df["YEAR"] <= 2013)
    ].copy()

    return df


df = load_data()

st.success("Dataset loaded successfully.")


# =========================================================
# DATA PREVIEW
# =========================================================

with st.expander("Preview the data"):
    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =========================================================
# MONTHLY TIME SERIES
# =========================================================

monthly = (
    df.groupby("DATE")
      .agg(
          Total_Cases=("CASES", "sum"),
          Reporting_Records=("CASES", "count")
      )
      .sort_index()
)


# Complete monthly calendar
full_dates = pd.date_range(
    start="2011-01-01",
    end="2013-12-01",
    freq="MS"
)

monthly = monthly.reindex(full_dates)

monthly.index.name = "DATE"


# December 2011 has no observation in the source data.
# It is intentionally retained as missing rather than
# being interpreted as zero reported cases.


# =========================================================
# ROLLING TREND
# =========================================================

monthly["Rolling_12M"] = (
    monthly["Total_Cases"]
    .rolling(
        window=12,
        min_periods=6
    )
    .mean()
)


# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.header("Time Series Controls")

resolution = st.sidebar.selectbox(
    "Select temporal resolution:",
    [
        "Monthly",
        "Quarterly",
        "Yearly"
    ]
)


# =========================================================
# TEMPORAL RESAMPLING
# =========================================================

if resolution == "Monthly":

    plot_data = monthly["Total_Cases"].copy()

    resolution_note = (
        "Monthly resolution preserves short-term variation "
        "and makes seasonal patterns visible."
    )


elif resolution == "Quarterly":

    plot_data = (
        monthly["Total_Cases"]
        .resample("QS")
        .sum(min_count=1)
    )

    resolution_note = (
        "Quarterly aggregation reduces month-to-month variation "
        "and emphasizes broader temporal patterns."
    )


else:

    plot_data = (
        monthly["Total_Cases"]
        .resample("YS")
        .sum(min_count=1)
    )

    resolution_note = (
        "Yearly aggregation provides the broadest temporal view, "
        "but within-year seasonal variation is hidden."
    )


# =========================================================
# RESOLUTION VISUALIZATION
# =========================================================

st.subheader("Temporal Resolution")

st.info(resolution_note)

st.subheader(
    f"{resolution} Reported LSD Cases"
)

fig1, ax1 = plt.subplots(
    figsize=(12, 5)
)

ax1.plot(
    plot_data.index,
    plot_data.values,
    marker="o",
    linewidth=2
)

ax1.set_title(
    f"{resolution} Lumpy Skin Disease Cases "
    "in Cattle, Zimbabwe (2011–2013)"
)

ax1.set_xlabel("Date")
ax1.set_ylabel("Total Reported Cases")

ax1.grid(alpha=0.3)


# Format x-axis according to resolution
if resolution == "Monthly":

    ax1.xaxis.set_major_locator(
        mdates.MonthLocator(interval=3)
    )

    ax1.xaxis.set_major_formatter(
        mdates.DateFormatter("%b\n%Y")
    )


elif resolution == "Quarterly":

    ax1.xaxis.set_major_locator(
        mdates.MonthLocator(interval=3)
    )

    ax1.xaxis.set_major_formatter(
        mdates.DateFormatter("%b\n%Y")
    )


else:

    ax1.xaxis.set_major_locator(
        mdates.YearLocator()
    )

    ax1.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )


fig1.tight_layout()

st.pyplot(fig1)

plt.close(fig1)


# =========================================================
# MONTHLY-ONLY ANALYSIS
# =========================================================

if resolution == "Monthly":

    # =====================================================
    # SEASONAL COMPARISON
    # =====================================================

    st.subheader(
        "Seasonal Pattern of Lumpy Skin Disease"
    )

    st.write(
        "The chart compares monthly reported LSD cases "
        "across 2011, 2012, and 2013 to examine recurring "
        "within-year patterns."
    )


    seasonal = (
        df.groupby(
            ["YEAR", "MONTH_NUM"]
        )["CASES"]
        .sum()
        .reset_index()
    )


    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec"
    ]


    fig2, ax2 = plt.subplots(
        figsize=(12, 5)
    )


    for year in [2011, 2012, 2013]:

        year_data = seasonal[
            seasonal["YEAR"] == year
        ]

        ax2.plot(
            year_data["MONTH_NUM"],
            year_data["CASES"],
            marker="o",
            linewidth=2,
            label=str(year)
        )


    ax2.set_title(
        "Seasonal Pattern of Monthly "
        "Lumpy Skin Disease Cases, 2011–2013"
    )

    ax2.set_xlabel("Month")

    ax2.set_ylabel(
        "Total Reported Cases"
    )

    ax2.set_xticks(
        range(1, 13)
    )

    ax2.set_xticklabels(
        month_names
    )

    ax2.legend(
        title="Year"
    )

    ax2.grid(
        alpha=0.3
    )

    fig2.tight_layout()

    st.pyplot(fig2)

    plt.close(fig2)


    # =====================================================
    # AVERAGE MONTHLY SEASONAL PROFILE
    # =====================================================

    st.subheader(
        "Average Monthly Pattern"
    )


    monthly_average = (
        seasonal
        .groupby("MONTH_NUM")["CASES"]
        .mean()
        .reindex(range(1, 13))
    )


    fig3, ax3 = plt.subplots(
        figsize=(12, 5)
    )


    ax3.bar(
        range(1, 13),
        monthly_average.values
    )


    ax3.set_title(
        "Average Monthly Lumpy Skin Disease "
        "Cases, 2011–2013"
    )

    ax3.set_xlabel(
        "Month"
    )

    ax3.set_ylabel(
        "Average Reported Cases"
    )

    ax3.set_xticks(
        range(1, 13)
    )

    ax3.set_xticklabels(
        month_names
    )

    ax3.grid(
        axis="y",
        alpha=0.3
    )


    fig3.tight_layout()

    st.pyplot(fig3)

    plt.close(fig3)


    st.info(
        "The seasonal profile shows the highest average "
        "reported LSD cases during March and April, followed "
        "by a substantial decline during the middle of the "
        "year. The pattern suggests strong seasonality in "
        "reported cases during the 2011–2013 period."
    )


    # =====================================================
    # TREND AND UNCERTAINTY
    # =====================================================

    st.subheader(
        "Trend and Uncertainty"
    )

    st.write(
        "The 12-month rolling mean summarizes the broader "
        "temporal pattern. The shaded band represents "
        "variability around the rolling mean."
    )


    rolling = (
        monthly["Total_Cases"]
        .rolling(
            window=12,
            min_periods=6
        )
    )


    rolling_mean = rolling.mean()

    rolling_std = rolling.std()


    lower_band = (
        rolling_mean -
        (2 * rolling_std)
    )

    upper_band = (
        rolling_mean +
        (2 * rolling_std)
    )


    fig4, ax4 = plt.subplots(
        figsize=(12, 5)
    )


    ax4.plot(
        monthly.index,
        monthly["Total_Cases"],
        marker="o",
        alpha=0.4,
        label="Monthly Cases"
    )


    ax4.plot(
        rolling_mean.index,
        rolling_mean,
        linewidth=2.5,
        label="12-Month Rolling Mean"
    )


    ax4.fill_between(
        rolling_mean.index,
        lower_band,
        upper_band,
        alpha=0.2,
        label="±2 Rolling Standard Deviations"
    )


    ax4.set_title(
        "LSD Trend and Variability, 2011–2013"
    )

    ax4.set_xlabel(
        "Date"
    )

    ax4.set_ylabel(
        "Total Reported Cases"
    )


    ax4.xaxis.set_major_locator(
        mdates.MonthLocator(
            interval=3
        )
    )


    ax4.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%b\n%Y"
        )
    )


    ax4.legend()

    ax4.grid(
        alpha=0.3
    )


    fig4.tight_layout()

    st.pyplot(fig4)

    plt.close(fig4)


    st.caption(
        "The shaded region represents ±2 rolling standard "
        "deviations around the 12-month rolling mean. "
        "It communicates local variability in reported cases "
        "and should not be interpreted as a formal confidence interval."
    )


# =========================================================
# TEMPORAL HONESTY
# =========================================================

st.subheader(
    "Temporal Honesty and Data Limitations"
)

st.write(
    "December 2011 has no observation in the source dataset. "
    "The month is retained as missing rather than being assigned "
    "a value of zero because an absent record does not necessarily "
    "mean that zero disease cases occurred."
)

st.write(
    "The application also allows the same data to be viewed at "
    "monthly, quarterly, and yearly resolutions. Monthly resolution "
    "reveals seasonal variation, while aggregation to quarterly or "
    "yearly resolution smooths short-term variation and can hide "
    "important within-year patterns."
)

st.write(
    "The y-axis begins at zero so that differences in reported case "
    "counts are not visually exaggerated."
)


# =========================================================
# SUMMARY
# =========================================================

st.subheader(
    "Summary"
)

st.write(
    "Reported Lumpy Skin Disease cases show a recurring concentration "
    "during the early months of the year, particularly around March "
    "and April during the 2011–2013 analysis period. Cases generally "
    "decline during the middle months of the year. The resolution "
    "selector demonstrates how temporal aggregation changes the "
    "visual interpretation of the same underlying data."
)