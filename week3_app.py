import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("Battry and theft in chicago.csv")

st.title("Week 3: Chicago Crime Multivariate Analysis")
st.write("Exploring theft and battery patterns across Chicago Community Areas in 2019 using correlation analysis and Principal Component Analysis (PCA).")

df["Date"] = pd.to_datetime(df["Date"])

df_2019 = df[df["Year"] == 2019].copy()

df_2019["Theft"] = (df_2019["Primary Type"] == "THEFT").astype(int)
df_2019["Battery"] = (df_2019["Primary Type"] == "BATTERY").astype(int)
df_2019["Arrest_Num"] = df_2019["Arrest"].astype(int)
df_2019["Domestic_Num"] = df_2019["Domestic"].astype(int)

df_2019["Night"] = (
    (df_2019["Date"].dt.hour >= 18) |
    (df_2019["Date"].dt.hour < 6)
).astype(int)

df_2019["Weekend"] = (df_2019["Date"].dt.dayofweek >= 5).astype(int)

community = df_2019.groupby("Community Area").agg(
    Total_Crimes=("ID", "count"),
    Theft_Count=("Theft", "sum"),
    Battery_Count=("Battery", "sum"),
    Arrest_Rate=("Arrest_Num", "mean"),
    Domestic_Rate=("Domestic_Num", "mean"),
    Night_Rate=("Night", "mean"),
    Weekend_Rate=("Weekend", "mean")
).reset_index()

st.subheader("Aggregated Data Preview")
st.dataframe(community.head())

features = [
    "Total_Crimes",
    "Theft_Count",
    "Battery_Count",
    "Arrest_Rate",
    "Domestic_Rate",
    "Night_Rate",
    "Weekend_Rate"
]

st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(9, 7))

sns.heatmap(
    community[features].corr(),
    annot=True,
    cmap="vlag",
    center=0,
    ax=ax
)

ax.set_title("Correlation of Chicago Crime Characteristics by Community Area (2019)")

st.pyplot(fig)

# Standardize the seven numeric features before PCA
X = community[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce the seven features to two principal components
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
pca_df["Community Area"] = community["Community Area"].values
pca_df["Total Crimes"] = community["Total_Crimes"].values

# Create a meaningful categorical variable for coloring
pca_df["Crime Level"] = pd.qcut(
    pca_df["Total Crimes"],
    q=3,
    labels=["Low", "Medium", "High"]
)
st.subheader("PCA Projection")

color_by = st.selectbox(
    "Color PCA points by:",
    ["Crime Level", "Total Crimes"]
)

fig2, ax2 = plt.subplots(figsize=(10, 7))

if color_by == "Crime Level":
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Crime Level",
        hue_order=["Low", "Medium", "High"],
        s=80,
        ax=ax2
    )
    ax2.legend(title="Crime Level")
else:
    scatter = ax2.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=pca_df["Total Crimes"],
        cmap="viridis",
        s=80
    )
    fig2.colorbar(scatter, ax=ax2, label="Total Crimes")

ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax2.set_title("PCA of Chicago Theft and Battery Characteristics by Community Area (2019)")
ax2.axhline(0, color="gray", linewidth=0.5)
ax2.axvline(0, color="gray", linewidth=0.5)

st.pyplot(fig2)

st.subheader("Interpretation")

st.write("""
The correlation heatmap shows strong positive relationships between total crime,
theft, and battery counts across Chicago Community Areas. Total crime is especially
strongly correlated with theft and battery counts.

PC1 explains 36.1% of the variation and primarily represents overall crime volume,
with total crimes, theft, and battery contributing most strongly to this component.
Community Areas with higher crime levels therefore tend to appear farther to the
right of the PCA plot.

PC2 explains 25.5% of the variation and mainly reflects differences in crime context,
particularly domestic, nighttime, arrest, and weekend rates. Together, PC1 and PC2
explain approximately 61.6% of the variation in the seven standardized crime
characteristics.
""")