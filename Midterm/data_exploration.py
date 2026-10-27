from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).parent / "buffalo_range_weather.csv"

with open(DATA_PATH, "r", encoding="utf-8-sig") as file:
    lines = file.readlines()

header_index = next(
    i for i, line in enumerate(lines)
    if line.strip().startswith("YEAR,")
)

df = pd.read_csv(DATA_PATH, skiprows=header_index)

print("\nDATASET SHAPE:", df.shape)
print("\nCOLUMNS:", df.columns.tolist())
print("\nFIRST FIVE ROWS:")
print(df.head())
print("\nMISSING VALUES (-999):")
print((df == -999).sum())

# Create a date column from YEAR and DOY
df["DATE"] = pd.to_datetime(
    df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3),
    format="%Y%j"
)

print("\n========== DATE COVERAGE ==========")
print("First date:", df["DATE"].min())
print("Last date:", df["DATE"].max())
print("Total records:", len(df))
print("Duplicate dates:", df["DATE"].duplicated().sum())

# --------------------------------------------------
# STEP 6: WEATHER VARIABLE STATISTICS
# --------------------------------------------------

weather_columns = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "WS10M",
    "WS10M_MAX",
    "WD10M",
    "PRECTOTCORR",
    "RH2M",
    "PS"
]

print("\n========== WEATHER STATISTICS ==========")

print(
    df[weather_columns]
    .describe()
    .round(2)
    .to_string()
)

print("\n========== NEGATIVE RAINFALL CHECK ==========")
print("Negative rainfall records:",
      (df["PRECTOTCORR"] < 0).sum())

# --------------------------------------------------
# STEP 7: EXTREME WEATHER EVENTS
# --------------------------------------------------

extreme_variables = {
    "Hottest day": "T2M_MAX",
    "Strongest wind day": "WS10M_MAX",
    "Wettest day": "PRECTOTCORR",
    "Highest humidity day": "RH2M"
}

print("\n========== EXTREME WEATHER EVENTS ==========")

for event, column in extreme_variables.items():
    idx = df[column].idxmax()
    date = df.loc[idx, "DATE"].strftime("%Y-%m-%d")
    value = df.loc[idx, column]

    print(f"{event}: {date} | {value:.2f}")