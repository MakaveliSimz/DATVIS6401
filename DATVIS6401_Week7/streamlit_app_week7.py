
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text
from wordcloud import WordCloud


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="20 Newsgroups | Text Explorer",
    page_icon="📰",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

st.title("📰 20 Newsgroups — Interactive Text Explorer")
st.caption("DATS 6401 | Week 7 — Text Data Visualization")


# --------------------------------------------------
# LOAD DATA AND PRECOMPUTED TOPICS
# --------------------------------------------------

@st.cache_data(show_spinner="Loading 20 Newsgroups corpus...")
def load_data():
    news = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes")
    )
    return news.data, news.target, news.target_names


@st.cache_data
def load_topics():
    with open(BASE_DIR / "week7_topics.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_umap():
    with open(BASE_DIR / "week7_umap.json", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


docs, labels, categories = load_data()

extra_stopwords = {
    "ax", "don", "ve", "does", "just", "like",
    "know", "think", "people", "use", "good",
    "make", "say", "way", "time"
}

stopwords = list(text.ENGLISH_STOP_WORDS.union(extra_stopwords))


# --------------------------------------------------
# APP TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Corpus and Counts",
    "🧠 BERTopic Explorer",
    "🗺️ UMAP Projection",
    "📖 Interpretation and Limitations"
])


# --------------------------------------------------
# TAB 1 — FREQUENCY, WORD CLOUD, TF-IDF
# --------------------------------------------------

with tab1:

    st.header("Exploring Word Frequencies")

    st.write(
        "This section compares two visual encodings of the same "
        "word-frequency data: a sorted bar chart and a word cloud."
    )

    selected_category = st.selectbox(
        "Choose a newsgroup category",
        ["All categories"] + list(categories)
    )

    if selected_category == "All categories":
        selected_docs = docs
    else:
        category_id = list(categories).index(selected_category)

        selected_docs = [
            doc
            for doc, label in zip(docs, labels)
            if label == category_id
        ]

    st.metric("Documents selected", f"{len(selected_docs):,}")

    max_words = st.slider(
        "Number of words to display",
        min_value=5,
        max_value=30,
        value=20
    )

    # Transform text into a document-term count matrix
    cv = CountVectorizer(
        stop_words=stopwords,
        max_features=max_words,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
    )

    matrix = cv.fit_transform(selected_docs)

    frequency_df = pd.DataFrame({
        "Word": cv.get_feature_names_out(),
        "Frequency": matrix.sum(axis=0).A1
    })

    frequency_df = frequency_df.sort_values("Frequency")

    # Sorted frequency bar chart
    fig_freq = px.bar(
        frequency_df,
        x="Frequency",
        y="Word",
        orientation="h",
        title="Top Words by Frequency",
        labels={
            "Frequency": "Total Word Frequency",
            "Word": "Word"
        }
    )

    fig_freq.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=60, b=20)
    )

    # Word cloud uses the SAME frequencies
    word_frequencies = dict(
        zip(
            frequency_df["Word"],
            frequency_df["Frequency"]
        )
    )

    cloud = WordCloud(
        width=900,
        height=550,
        background_color="white",
        colormap="viridis",
        max_words=max_words,
        random_state=42
    ).generate_from_frequencies(word_frequencies)

    fig_cloud, ax = plt.subplots(figsize=(9, 5.5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")

    # Display both visualizations side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Sorted Frequency Chart")
        st.plotly_chart(fig_freq, use_container_width=True)

    with col2:
        st.subheader("☁️ Word Cloud")
        st.pyplot(fig_cloud)

    plt.close(fig_cloud)

    st.info(
        "Both visualizations represent exactly the same word counts. "
        "The sorted bar chart makes comparisons more accurate. "
        "The word cloud is visually engaging, but font sizes and "
        "positions make precise comparisons difficult."
    )

    # --------------------------------------------------
    # TF-IDF COMPARISON
    # --------------------------------------------------

    st.divider()

    st.header("TF-IDF — A Different View of Important Terms")

    st.write(
        "Raw frequency counts how often words appear. "
        "TF-IDF weights terms according to their frequency "
        "within documents and their rarity across documents."
    )

    tfidf = TfidfVectorizer(
        stop_words=stopwords,
        max_features=1000,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
    )

    tfidf_matrix = tfidf.fit_transform(selected_docs)

    tfidf_df = pd.DataFrame({
        "Word": tfidf.get_feature_names_out(),
        "Mean TF-IDF": tfidf_matrix.mean(axis=0).A1
    })

    tfidf_df = (
        tfidf_df
        .nlargest(max_words, "Mean TF-IDF")
        .sort_values("Mean TF-IDF")
    )

    fig_tfidf = px.bar(
        tfidf_df,
        x="Mean TF-IDF",
        y="Word",
        orientation="h",
        title="Top Terms by Average TF-IDF",
        labels={
            "Mean TF-IDF": "Mean TF-IDF Score",
            "Word": "Term"
        }
    )

    fig_tfidf.update_layout(height=600)

    st.plotly_chart(fig_tfidf, use_container_width=True)

    st.warning(
        "TF-IDF scores are not probabilities. High average TF-IDF "
        "does not necessarily mean a word represents a meaningful topic. "
        "The results depend on the corpus and preprocessing choices."
    )


# --------------------------------------------------
# TAB 2 — BERTOPIC EXPLORER
# --------------------------------------------------

with tab2:

    st.header("🧠 BERTopic — Discovering Themes in Text")

    st.write(
        "BERTopic uses document embeddings, dimensionality reduction "
        "and clustering to discover groups of semantically related "
        "documents. Representative words summarize each group."
    )

    st.caption(
        "The model was trained locally on a sample of 300 documents. "
        "Results were exported to JSON so the Streamlit app does not "
        "need to run the computationally expensive model."
    )

    topic_data = load_topics()

    topic_options = {
        (
            f"Topic {t['topic_id']} — "
            f"{', '.join(w['word'] for w in t['words'][:3])}"
        ): t
        for t in topic_data
    }

    selected_name = st.selectbox(
        "Choose a discovered topic",
        list(topic_options.keys())
    )

    selected_topic = topic_options[selected_name]

    st.metric(
        "Documents assigned to selected topic",
        selected_topic["count"]
    )

    words_df = pd.DataFrame(selected_topic["words"])
    words_df = words_df.sort_values("score")

    fig_topic = px.bar(
        words_df,
        x="score",
        y="word",
        orientation="h",
        title=(
            f"Topic {selected_topic['topic_id']} "
            "— Representative Terms"
        ),
        labels={
            "score": "BERTopic c-TF-IDF Score",
            "word": "Representative Word"
        }
    )

    fig_topic.update_layout(height=500)

    st.plotly_chart(fig_topic, use_container_width=True)

    st.divider()

    st.subheader("Topic Size Comparison")

    sizes_df = pd.DataFrame([
        {
            "Topic": f"Topic {t['topic_id']}",
            "Documents": t["count"]
        }
        for t in topic_data
    ])

    fig_sizes = px.bar(
        sizes_df,
        x="Topic",
        y="Documents",
        title="Documents Assigned to Each Topic"
    )

    st.plotly_chart(fig_sizes, use_container_width=True)

    st.info(
        "The model discovered six non-outlier topics in the "
        "sample. Topic -1 represents documents classified as "
        "outliers and is not included in this chart."
    )

    st.warning(
        "Topic models do not discover objectively fixed categories. "
        "Their results can change with sampling, preprocessing, "
        "embedding models and clustering parameters."
    )


# --------------------------------------------------
# TAB 3 — INTERPRETATION
# --------------------------------------------------

with tab3:
    st.header("🗺️ UMAP — Document Embedding Projection")
    st.write(
        "Each point represents one of 300 sampled news documents. "
        "Colors indicate BERTopic assignments, and hovering reveals a text preview."
    )
    umap_df = load_umap()
    umap_df["Topic"] = umap_df["topic"].apply(
        lambda value: "Outliers (-1)" if int(value) == -1 else f"Topic {int(value)}"
    )
    all_topics = sorted(umap_df["Topic"].unique(),
                        key=lambda value: -1 if value == "Outliers (-1)" else int(value.split()[-1]))
    chosen_topics = st.multiselect(
        "Topics to display", all_topics, default=all_topics
    )
    filtered = umap_df[umap_df["Topic"].isin(chosen_topics)].copy()
    st.metric("Documents displayed", len(filtered))
    if filtered.empty:
        st.info("Select at least one topic to display its documents.")
    else:
        fig_umap = px.scatter(
            filtered, x="x", y="y", color="Topic",
            hover_data={"preview": True, "document_id": True, "x": False, "y": False},
            title="UMAP Projection of 300 News Documents",
            labels={"x": "UMAP dimension 1", "y": "UMAP dimension 2"},
            opacity=0.8,
        )
        fig_umap.update_traces(marker={"size": 10})
        fig_umap.update_layout(height=650, legend_title_text="BERTopic group")
        st.plotly_chart(fig_umap, use_container_width=True)
    st.caption(
        "UMAP compresses high-dimensional embeddings into two dimensions. "
        "Apparent distances and clusters are approximate and depend on projection settings. "
        "Topic -1 denotes outliers. This projection uses the saved 300-document sample."
    )


# --------------------------------------------------
# TAB 4 — INTERPRETATION
# --------------------------------------------------

with tab4:

    st.header("📖 Interpretation and Visualization Limitations")

    st.subheader("1. Sorted Word Frequency Chart")

    st.write(
        "**What it reveals:** The most frequently occurring words "
        "and their relative counts across the selected documents."
    )

    st.write(
        "**How it can mislead:** Frequent words may be uninformative "
        "or reflect artifacts in the dataset rather than important themes."
    )

    st.subheader("2. Word Cloud")

    st.write(
        "**What it reveals:** A quick visual impression of commonly "
        "occurring terms."
    )

    st.write(
        "**How it can mislead:** Font size, position and layout make "
        "accurate comparisons difficult. The layout does not represent "
        "semantic similarity."
    )

    st.subheader("3. TF-IDF")

    st.write(
        "**What it reveals:** Terms weighted by their frequency "
        "within documents and rarity across the corpus."
    )

    st.write(
        "**How it can mislead:** High scores do not automatically "
        "identify meaningful subjects. Different preprocessing "
        "settings can change the rankings."
    )

    st.subheader("4. BERTopic")

    st.write(
        "**What it reveals:** Clusters of documents summarized "
        "by representative terms, helping reveal recurring themes "
        "that individual word counts may miss."
    )

    st.write(
        "**How it can mislead:** Topics can mix unrelated subjects, "
        "overlap or change when the sample and model parameters change."
    )

    st.subheader("5. Embedding Projection")

    st.write(
        "A two-dimensional projection can visualize patterns "
        "of document similarity."
    )

    st.write(
        "**Limitation:** A 2D projection compresses high-dimensional "
        "information. Distances on the plot are not exact measures "
        "of semantic similarity."
    )

    st.caption(
        "The optional UMAP projection is available in the UMAP Projection tab, "
        "using the 300-document sample exported from the notebook."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "DATS 6401 — Visualization of Complex Data | Week 7 | "
    "Dataset: scikit-learn 20 Newsgroups | "
    "Methods: CountVectorizer, TF-IDF, BERTopic and UMAP | "
    "Built with Streamlit, Plotly and Matplotlib"
)
