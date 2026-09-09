import streamlit as st
import plotly.express as px

st.title("Week 2: Visualization of Complex Data")
st.write("My first Streamlit data visualization app.")

df = px.data.gapminder()
df_2007 = df[df["year"] == 2007]
continent_pop = df_2007.groupby("continent")["pop"].sum().reset_index()

st.subheader("Dataset Preview")
st.dataframe(df.head())
st.subheader("Population by Continent - 2007")
st.dataframe(continent_pop)

st.subheader("Question")
st.write("How is the world population distributed across continents in 2007?")

fig_pie = px.pie(
    continent_pop,
    names="continent",
    values="pop",
    title="Share of World Population by Continent (2007)"
)

st.plotly_chart(fig_pie, use_container_width=True)

fig_bar = px.bar(
    continent_pop,
    x="continent",
    y="pop",
    title="Population by Continent (2007)",
    labels={
        "continent": "Continent",
        "pop": "Population"
    }
)

st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Interpretation")

st.write("""
The charts show that Asia had the largest share of the population in the 2007 Gapminder dataset.
Africa and the Americas had similar population totals, while Oceania had the smallest population.
The pie chart is useful for showing proportions, while the bar chart makes comparisons between continents easier.
""")
