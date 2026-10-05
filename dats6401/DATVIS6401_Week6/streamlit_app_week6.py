import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Zimbabwe Airport Route Network",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Zimbabwe Airport Route Network")
st.subheader("Interactive Graph and Network Analysis")

st.write(
    """
    Explore air routes involving Zimbabwean airports.
    Airports are shown as circles (nodes), while direct air routes
    are shown as lines (edges).

    Hover over airports and routes to explore the network.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data_folder = Path(__file__).parent

    airport_columns = [
        "Airport_ID", "Name", "City", "Country",
        "IATA", "ICAO", "Latitude", "Longitude",
        "Altitude", "Timezone", "DST",
        "Tz_database", "Type", "Source"
    ]

    route_columns = [
        "Airline", "Airline_ID",
        "Source_Airport", "Source_Airport_ID",
        "Destination_Airport", "Destination_Airport_ID",
        "Codeshare", "Stops", "Equipment"
    ]

    airline_columns = [
        "Airline_ID", "Name", "Alias",
        "IATA", "ICAO", "Callsign",
        "Country", "Active"
    ]

    airports = pd.read_csv(
        data_folder / "airports.dat",
        header=None,
        names=airport_columns
    )

    routes = pd.read_csv(
        data_folder / "routes.dat",
        header=None,
        names=route_columns
    )

    airlines = pd.read_csv(
        data_folder / "airlines.dat",
        header=None,
        names=airline_columns
    )

    return airports, routes, airlines


airports, routes, airlines = load_data()


# ============================================================
# CLEAN DATA
# ============================================================

airports["IATA"] = airports["IATA"].replace("\\N", pd.NA)
airlines["IATA"] = airlines["IATA"].replace("\\N", pd.NA)

zimbabwe_airports = airports[
    airports["Country"].str.strip().str.lower() == "zimbabwe"
].copy()

zim_iata = set(
    zimbabwe_airports["IATA"].dropna()
)

zim_routes = routes[
    routes["Source_Airport"].isin(zim_iata)
    |
    routes["Destination_Airport"].isin(zim_iata)
].copy()


# ============================================================
# LOOKUP TABLES
# ============================================================

airport_lookup = (
    airports
    .dropna(subset=["IATA"])
    .drop_duplicates(subset=["IATA"])
    .set_index("IATA")
)

airline_lookup = (
    airlines
    .dropna(subset=["IATA"])
    .drop_duplicates(subset=["IATA"])
    .set_index("IATA")["Name"]
    .to_dict()
)


# ============================================================
# BUILD NETWORK
# ============================================================

G = nx.DiGraph()

for _, row in zim_routes.iterrows():

    source = row["Source_Airport"]
    destination = row["Destination_Airport"]

    if pd.notna(source) and pd.notna(destination):

        G.add_edge(
            source,
            destination
        )


# ============================================================
# NETWORK METRICS
# ============================================================

degree_centrality = nx.degree_centrality(G)

betweenness_centrality = nx.betweenness_centrality(G)


network_metrics = pd.DataFrame(
    {
        "Airport": list(G.nodes()),
        "Degree Centrality": [
            degree_centrality[node]
            for node in G.nodes()
        ],
        "Betweenness Centrality": [
            betweenness_centrality[node]
            for node in G.nodes()
        ]
    }
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def airport_name_from_code(code):

    if code in airport_lookup.index:

        row = airport_lookup.loc[code]

        return (
            f"{row['Name']} ({code}), "
            f"{row['City']}, {row['Country']}"
        )

    return code


def get_airport_details(code):

    code = code.upper().strip()

    if code not in G.nodes:
        return None

    if code in airport_lookup.index:

        row = airport_lookup.loc[code]

        name = row["Name"]
        city = row["City"]
        country = row["Country"]

    else:

        name = "Unknown Airport"
        city = "Unknown"
        country = "Unknown"

    connected = sorted(
        set(G.predecessors(code))
        |
        set(G.successors(code))
    )

    airport_routes = zim_routes[
        (zim_routes["Source_Airport"] == code)
        |
        (zim_routes["Destination_Airport"] == code)
    ]

    airline_codes = (
        airport_routes["Airline"]
        .dropna()
        .astype(str)
        .unique()
    )

    airline_names = sorted(
        set(
            airline_lookup.get(
                airline_code,
                airline_code
            )
            for airline_code in airline_codes
        )
    )

    return {
        "code": code,
        "name": name,
        "city": city,
        "country": country,
        "connected": connected,
        "incoming": G.in_degree(code),
        "outgoing": G.out_degree(code),
        "airlines": airline_names,
        "degree": degree_centrality.get(code, 0),
        "betweenness": betweenness_centrality.get(code, 0)
    }


def find_airport_codes(question):

    words = (
        question
        .upper()
        .replace("?", " ")
        .replace(",", " ")
        .replace(".", " ")
        .split()
    )

    return [
        code
        for code in G.nodes()
        if code.upper() in words
    ]


def get_route_airlines(source, destination):

    rows = zim_routes[
        (
            (zim_routes["Source_Airport"] == source)
            &
            (zim_routes["Destination_Airport"] == destination)
        )
        |
        (
            (zim_routes["Source_Airport"] == destination)
            &
            (zim_routes["Destination_Airport"] == source)
        )
    ]

    if rows.empty:
        return []

    codes = (
        rows["Airline"]
        .dropna()
        .astype(str)
        .unique()
    )

    return sorted(
        set(
            airline_lookup.get(code, code)
            for code in codes
        )
    )


# ============================================================
# SIMBA RESPONSE ENGINE
# ============================================================

def simba_answer(question):

    q = question.lower().strip()

    airport_codes = find_airport_codes(question)

    # CENTRALITY EXPLANATION
    if (
        "degree centrality" in q
        or
        "direct connectivity" in q
    ):

        return (
            "📊 **Direct Connectivity (Degree Centrality)** tells you "
            "how strongly an airport is directly connected to other "
            "airports in this network. A higher value means more "
            "direct connectivity."
        )

    if (
        "betweenness" in q
        or
        "bridge importance" in q
    ):

        return (
            "🌉 **Bridge Importance (Betweenness Centrality)** shows "
            "how important an airport is in connecting different parts "
            "of the network. A high value means many connecting paths "
            "pass through that airport."
        )

    # MOST CONNECTED
    if (
        "most connected" in q
        or
        "main hub" in q
        or
        "biggest hub" in q
    ):

        top_airport = max(
            degree_centrality,
            key=degree_centrality.get
        )

        details = get_airport_details(top_airport)

        return (
            f"🏆 **{details['name']} ({top_airport})** in "
            f"**{details['city']}, {details['country']}** is the "
            f"most connected airport in this network, with "
            f"**{len(details['connected'])} directly connected airports**."
        )

    # STRONGEST BRIDGE
    if (
        "strongest bridge" in q
        or
        "most important bridge" in q
        or
        "highest betweenness" in q
    ):

        top_airport = max(
            betweenness_centrality,
            key=betweenness_centrality.get
        )

        details = get_airport_details(top_airport)

        return (
            f"🌉 **{details['name']} ({top_airport})** is the "
            f"strongest bridge in this network, with a Bridge "
            f"Importance score of **{details['betweenness']:.3f}**."
        )

    # TWO AIRPORTS
    if len(airport_codes) >= 2:

        source = airport_codes[0]
        destination = airport_codes[1]

        route_airlines = get_route_airlines(
            source,
            destination
        )

        if route_airlines:

            airline_text = ", ".join(route_airlines)

            return (
                f"✈️ The route between "
                f"**{airport_name_from_code(source)}** and "
                f"**{airport_name_from_code(destination)}** is "
                f"represented in this dataset by: "
                f"**{airline_text}**."
            )

        return (
            f"🦁 I cannot find a direct route between "
            f"**{source}** and **{destination}** in this dataset."
        )

    # SINGLE AIRPORT
    if len(airport_codes) == 1:

        code = airport_codes[0]

        details = get_airport_details(code)

        if details:

            # AIRLINES
            if (
                "airline" in q
                or
                "serve" in q
            ):

                if details["airlines"]:

                    airline_text = ", ".join(
                        details["airlines"]
                    )

                    return (
                        f"🛫 Airlines represented for "
                        f"**{details['name']} ({code})** include: "
                        f"**{airline_text}**."
                    )

            # DESTINATIONS
            if (
                "where can i fly" in q
                or
                "destinations" in q
                or
                "connected to" in q
            ):

                destinations = sorted(
                    set(G.successors(code))
                )

                if destinations:

                    destination_names = [
                        airport_name_from_code(destination)
                        for destination in destinations
                    ]

                    text = "\n\n- ".join(
                        destination_names
                    )

                    return (
                        f"🌍 Direct outbound destinations represented "
                        f"from **{details['name']} ({code})** are:"
                        f"\n\n- {text}"
                    )

            # CENTRALITY FOR SPECIFIC AIRPORT
            if (
                "centrality" in q
                or
                "importance" in q
                or
                "score" in q
            ):

                return (
                    f"📊 **{details['name']} ({code})** has a "
                    f"Direct Connectivity score of "
                    f"**{details['degree']:.3f}** and a Bridge "
                    f"Importance score of "
                    f"**{details['betweenness']:.3f}**."
                )

            # GENERAL AIRPORT QUESTION
            return (
                f"✈️ **{details['name']} ({code})** is located in "
                f"**{details['city']}, {details['country']}**. "
                f"In this network, it has "
                f"**{len(details['connected'])} directly connected "
                f"airport(s)**."
            )

    # NUMBER OF AIRPORTS
    if (
        "how many airports" in q
        or
        "number of airports" in q
    ):

        return (
            f"🌍 This network contains "
            f"**{G.number_of_nodes()} airports**."
        )

    # NUMBER OF ROUTES
    if (
        "how many routes" in q
        or
        "number of routes" in q
        or
        "route connections" in q
    ):

        return (
            f"✈️ This network contains "
            f"**{G.number_of_edges()} directed route connections**."
        )

    # DATA SOURCE
    if (
        "data source" in q
        or
        "where is the data from" in q
        or
        "source of data" in q
    ):

        return (
            "📚 This dashboard uses the **OpenFlights airports, "
            "routes and airlines datasets**. The results describe "
            "the network represented in those datasets."
        )

    # FALLBACK / ESCALATION
    return (
        "🦁 I'm sorry, I don't have enough information in the "
        "current aviation network dataset to answer that question."
        "\n\n"
        "For further assistance, kindly contact the "
        "**Manager, Aeronautical Information Management (AIM)** "
        "at **+263 778 034 819** or email **aimzim@ac.zw**."
    )


# ============================================================
# ADD AIRPORT INFORMATION TO TABLE
# ============================================================

def table_airport_information(code):

    if code in airport_lookup.index:

        row = airport_lookup.loc[code]

        return pd.Series(
            [
                row["Name"],
                row["City"],
                row["Country"]
            ]
        )

    return pd.Series(
        ["Unknown", "Unknown", "Unknown"]
    )


network_metrics[
    ["Airport Name", "City", "Country"]
] = network_metrics["Airport"].apply(
    table_airport_information
)


network_metrics = network_metrics[
    [
        "Airport",
        "Airport Name",
        "City",
        "Country",
        "Degree Centrality",
        "Betweenness Centrality"
    ]
].sort_values(
    "Degree Centrality",
    ascending=False
).reset_index(drop=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("✈️ Network Controls")


metric_choice = st.sidebar.selectbox(
    "What would you like to explore?",
    [
        "Direct Connectivity (Degree Centrality)",
        "Bridge Importance (Betweenness Centrality)"
    ]
)


selected_airport = st.sidebar.selectbox(
    "Select an airport:",
    ["None"] + sorted(G.nodes())
)


st.sidebar.markdown("---")

st.sidebar.subheader("How to read the network")

st.sidebar.write(
    """
    **Circle = Airport**

    **Line = Direct air route**

    🟢 Zimbabwe airport

    🔵 Foreign airport

    🟠 Selected airport

    **Bigger circle = greater importance under the selected measure**
    """
)


# ============================================================
# SELECT METRIC
# ============================================================

if metric_choice == "Direct Connectivity (Degree Centrality)":

    metric_values = degree_centrality
    plain_metric_name = "Direct Connectivity"

else:

    metric_values = betweenness_centrality
    plain_metric_name = "Bridge Importance"


# ============================================================
# NETWORK OVERVIEW
# ============================================================

st.header("Network Overview")


col1, col2, col3 = st.columns(3)


col1.metric(
    "Airports in Network",
    G.number_of_nodes()
)


col2.metric(
    "Route Connections",
    G.number_of_edges()
)


main_hub = max(
    degree_centrality,
    key=degree_centrality.get
)


col3.metric(
    "Most Connected Airport",
    main_hub
)


# ============================================================
# SIMBA QUICK HELP
# ============================================================

st.markdown(
    """
    <style>

    @keyframes simbaPulse {
        0% {
            transform: scale(1);
            opacity: 1;
        }

        50% {
            transform: scale(1.03);
            opacity: 0.75;
        }

        100% {
            transform: scale(1);
            opacity: 1;
        }
    }

    .simba-title {
        font-size: 24px;
        font-weight: 700;
        animation: simbaPulse 1.8s infinite;
    }

    </style>

    <div class="simba-title">
        🦁 Simba Quick Help ✨
    </div>
    """,
    unsafe_allow_html=True
)


st.caption(
    "Select a common question for an instant answer, "
    "or ask Simba your own question."
)


faq_questions = [
    "Select a frequently asked question...",
    "Which airport is the most connected?",
    "Which airport is the strongest network bridge?",
    "How many airports are in the network?",
    "How many route connections are in the network?",
    "Which airlines serve HRE?",
    "Where can I fly from HRE?",
    "Which airlines connect HRE and JNB?",
    "What does degree centrality mean?",
    "What does betweenness centrality mean?",
    "Where does the data come from?"
]


faq_choice = st.selectbox(
    "Frequently Asked Questions",
    faq_questions,
    key="simba_faq"
)


if faq_choice != "Select a frequently asked question...":

    st.markdown("#### 🦁 Simba")

    st.markdown(
        simba_answer(faq_choice)
    )


with st.expander(
    "💬 Ask Simba your own question"
):

    custom_question = st.text_input(
        "Type your question:",
        placeholder="Example: What is AMS?",
        key="simba_custom_question"
    )

    if custom_question:

        st.markdown("#### 🦁 Simba")

        st.markdown(
            simba_answer(custom_question)
        )


# ============================================================
# DASHBOARD TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🌐 Interactive Network",
        "📊 Airport Rankings",
        "🛫 Route Data"
    ]
)


# ============================================================
# TAB 1 — INTERACTIVE NETWORK
# ============================================================

with tab1:

    st.subheader(
        "Interactive Zimbabwe Airport Connectivity Network"
    )

    st.info(
        """
        **Move your cursor over an airport** to see its full name,
        city, country, connections and airlines.

        **Move your cursor over a route** to explore the airports,
        airlines and aircraft represented on that connection.
        """
    )


    pos = nx.spring_layout(
        G,
        seed=42
    )


    fig = go.Figure()


    # ========================================================
    # ROUTES
    # ========================================================

    route_groups = zim_routes.groupby(
        [
            "Source_Airport",
            "Destination_Airport"
        ],
        dropna=True
    )


    for (source, destination), group in route_groups:

        if source not in pos or destination not in pos:
            continue


        x0, y0 = pos[source]
        x1, y1 = pos[destination]


        if source in airport_lookup.index:

            source_row = airport_lookup.loc[source]

            source_name = source_row["Name"]
            source_city = source_row["City"]
            source_country = source_row["Country"]

        else:

            source_name = source
            source_city = "Unknown"
            source_country = "Unknown"


        if destination in airport_lookup.index:

            destination_row = airport_lookup.loc[destination]

            destination_name = destination_row["Name"]
            destination_city = destination_row["City"]
            destination_country = destination_row["Country"]

        else:

            destination_name = destination
            destination_city = "Unknown"
            destination_country = "Unknown"


        airline_codes = (
            group["Airline"]
            .dropna()
            .astype(str)
            .unique()
        )


        airline_names = sorted(
            set(
                airline_lookup.get(
                    code,
                    code
                )
                for code in airline_codes
            )
        )


        if airline_names:

            airline_text = "<br>".join(
                airline_names
            )

        else:

            airline_text = "Not available"


        equipment = (
            group["Equipment"]
            .dropna()
            .astype(str)
            .unique()
        )


        if len(equipment) > 0:

            equipment_text = ", ".join(
                equipment
            )

        else:

            equipment_text = "Not available"


        route_hover = (
            f"<b>{source_city} ({source}) → "
            f"{destination_city} ({destination})</b>"
            f"<br><br>"
            f"<b>From:</b> {source_name}"
            f"<br>{source_city}, {source_country}"
            f"<br><br>"
            f"<b>To:</b> {destination_name}"
            f"<br>{destination_city}, {destination_country}"
            f"<br><br>"
            f"<b>Airline(s):</b><br>{airline_text}"
            f"<br><br>"
            f"<b>Aircraft / Equipment:</b><br>{equipment_text}"
        )


        # Invisible line for easier hover
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    width=12,
                    color="rgba(0,0,0,0)"
                ),
                text=[
                    route_hover,
                    route_hover
                ],
                hoverinfo="text",
                showlegend=False
            )
        )


        # Visible route
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    width=1.5,
                    color="rgba(90,90,90,0.55)"
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )


    # ========================================================
    # AIRPORT NODES
    # ============================================================

    node_x = []
    node_y = []

    node_labels = []
    node_hover = []

    node_sizes = []
    node_colors = []


    for node in G.nodes():

        x, y = pos[node]

        node_x.append(x)
        node_y.append(y)

        node_labels.append(node)


        if node in airport_lookup.index:

            airport_row = airport_lookup.loc[node]

            airport_name = airport_row["Name"]
            city = airport_row["City"]
            country = airport_row["Country"]

        else:

            airport_name = node
            city = "Unknown"
            country = "Unknown"


        incoming = G.in_degree(node)
        outgoing = G.out_degree(node)


        connected_airports = len(
            set(G.predecessors(node))
            |
            set(G.successors(node))
        )


        airport_route_rows = zim_routes[
            (zim_routes["Source_Airport"] == node)
            |
            (zim_routes["Destination_Airport"] == node)
        ]


        airport_airline_codes = (
            airport_route_rows["Airline"]
            .dropna()
            .astype(str)
            .unique()
        )


        airport_airlines = sorted(
            set(
                airline_lookup.get(
                    code,
                    code
                )
                for code in airport_airline_codes
            )
        )


        if airport_airlines:

            airport_airline_text = "<br>".join(
                airport_airlines
            )

        else:

            airport_airline_text = "Not available"


        hover_text = (
            f"<b>{airport_name}</b>"
            f"<br>{city}, {country}"
            f"<br>IATA Code: {node}"
            f"<br><br>"
            f"<b>Directly Connected Airports:</b> "
            f"{connected_airports}"
            f"<br>"
            f"<b>Outgoing Routes:</b> {outgoing}"
            f"<br>"
            f"<b>Incoming Routes:</b> {incoming}"
            f"<br><br>"
            f"<b>{plain_metric_name} Score:</b> "
            f"{metric_values[node]:.3f}"
            f"<br><br>"
            f"<b>Airlines Represented:</b>"
            f"<br>{airport_airline_text}"
        )


        node_hover.append(
            hover_text
        )


        node_sizes.append(
            32
            +
            (
                metric_values[node]
                *
                40
            )
        )


        if (
            selected_airport != "None"
            and
            node == selected_airport
        ):

            node_colors.append(
                "orange"
            )

        elif node in zim_iata:

            node_colors.append(
                "green"
            )

        else:

            node_colors.append(
                "steelblue"
            )


    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_labels,
            textposition="middle center",
            hovertext=node_hover,
            hoverinfo="text",

            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(
                    width=2,
                    color="white"
                )
            ),

            textfont=dict(
                size=11,
                color="black"
            ),

            showlegend=False
        )
    )


    fig.update_layout(

        title=(
            "Zimbabwe Airport Route Network"
            f"<br><sup>"
            f"Circle size represents {plain_metric_name}"
            f"</sup>"
        ),

        hovermode="closest",

        showlegend=False,

        height=720,

        margin=dict(
            l=20,
            r=20,
            t=80,
            b=20
        ),

        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),

        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # SIMPLE INTERPRETATION
    # ========================================================

    st.subheader(
        "💡 What does this graph mean?"
    )


    if metric_choice == "Direct Connectivity (Degree Centrality)":

        st.write(
            """
            **Direct Connectivity** shows how strongly each airport
            is directly connected to other airports in this network.

            Larger circles represent airports with more direct
            connectivity. Harare (HRE) is the dominant connectivity
            hub in this dataset.
            """
        )

    else:

        st.write(
            """
            **Bridge Importance** shows which airports play an
            important role in connecting different parts of the network.

            Larger circles indicate stronger bridging importance.
            Harare (HRE) has the strongest bridging role in this dataset.
            """
        )


    # ========================================================
    # SELECTED AIRPORT
    # ========================================================

    if selected_airport != "None":

        st.subheader(
            "📍 Selected Airport"
        )

        details = get_airport_details(
            selected_airport
        )

        if details:

            st.success(
                f"✈️ **{details['name']} "
                f"({selected_airport})**  \n"
                f"📍 {details['city']}, "
                f"{details['country']}  \n"
                f"🔗 {len(details['connected'])} "
                f"directly connected airports"
            )

            if details["airlines"]:

                st.write(
                    "**Airlines represented in this dataset:**"
                )

                st.write(
                    ", ".join(
                        details["airlines"]
                    )
                )


# ============================================================
# TAB 2 — AIRPORT RANKINGS
# ============================================================

with tab2:

    st.subheader(
        "Airport Network Rankings"
    )

    st.write(
        """
        Compare airports using direct connectivity and bridge
        importance. Full airport names, cities and countries are
        included for easier interpretation.
        """
    )


    st.dataframe(
        network_metrics,
        use_container_width=True,
        hide_index=True
    )


    col1, col2 = st.columns(2)


    with col1:

        st.info(
            """
            **Direct Connectivity**

            Technical name: Degree Centrality.

            Shows how strongly an airport is directly connected
            to other airports in this network.
            """
        )


    with col2:

        st.info(
            """
            **Bridge Importance**

            Technical name: Betweenness Centrality.

            Shows how important an airport is in connecting
            different parts of the network.
            """
        )


# ============================================================
# TAB 3 — ROUTE DATA
# ============================================================

with tab3:

    st.subheader(
        "Routes Involving Zimbabwe"
    )


    route_display = zim_routes[
        [
            "Airline",
            "Source_Airport",
            "Destination_Airport",
            "Stops",
            "Equipment"
        ]
    ].copy()


    route_display["Airline Name"] = (
        route_display["Airline"]
        .map(airline_lookup)
        .fillna(route_display["Airline"])
    )


    route_display = route_display[
        [
            "Airline Name",
            "Airline",
            "Source_Airport",
            "Destination_Airport",
            "Stops",
            "Equipment"
        ]
    ]


    st.dataframe(
        route_display,
        use_container_width=True,
        hide_index=True
    )


    csv = route_display.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download Route Data as CSV",
        data=csv,
        file_name="zimbabwe_airport_routes.csv",
        mime="text/csv"
    )


# ============================================================
# DATA LIMITATIONS
# ============================================================

st.header("⚠️ Data Limitations")


st.warning(
    """
    This visualization describes the routes represented in the
    OpenFlights dataset used for this exercise.

    It should not be interpreted as a complete picture of current
    airline operations. Routes, airlines and aircraft may change
    over time, and some current or historical services may not be
    represented.
    """
)


# ============================================================
# KEY TAKEAWAY
# ============================================================

st.header("Key Takeaway")


st.write(
    """
    Within this dataset, Harare (HRE) is the dominant connectivity
    hub and network bridge, while Victoria Falls (VFA) plays a
    secondary connectivity role.

    The dashboard allows users to explore airport connections,
    airline information and network importance interactively.
    """
)