
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import Normalize

st.set_page_config(
    page_title="Week 8 | Handwritten Digits Explorer",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Handwritten Digits — Interactive Image Explorer")
st.caption("DATS 6401 | Week 8 — Image Data Visualization")
st.markdown(
    "Explore image collections, pixel intensities, classification behavior, "
    "and model explanations using the scikit-learn Digits dataset."
)


@st.cache_data
def load_data():
    digits = load_digits()
    return digits.images, digits.data, digits.target


@st.cache_resource
def train_model():
    images, X, y = load_data()
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=120, random_state=42, n_jobs=-1
    )
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
    accuracy = np.mean(predictions == y[test_idx])
    return model, test_idx, predictions, accuracy


images, X, labels = load_data()
model, test_idx, test_predictions, accuracy = train_model()

tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Image Montage",
    "📊 Image Structure",
    "🧠 Model & Saliency",
    "📖 Interpretation",
])

with tab1:
    st.header("Image Collection and Class Balance")
    st.write(
        "A montage makes variation in handwriting visible and helps identify "
        "unusual shapes, faint digits, and possible labeling issues."
    )

    digit = st.selectbox("Choose digit class", list(range(10)))
    count = st.slider("Number of images", 6, 36, 18, 6)

    selected = images[labels == digit][:count]
    columns = 6
    rows = int(np.ceil(len(selected) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(11, rows * 1.8))
    axes = np.asarray(axes).ravel()

    for ax, img in zip(axes, selected):
        ax.imshow(img, cmap="gray", vmin=0, vmax=16)
        ax.axis("off")
    for ax in axes[len(selected):]:
        ax.axis("off")

    fig.suptitle(f"Montage: Handwritten Digit {digit}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Class Balance")
    counts = np.bincount(labels, minlength=10)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(np.arange(10), counts, color="#4179a8")
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit class")
    ax.set_ylabel("Number of images")
    ax.set_title("Images per digit class")
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        f"Dataset contains {len(images):,} images, each 8×8 pixels. "
        f"Digit {digit} has {counts[digit]} images. "
        "The classes are approximately balanced, although handwriting "
        "styles vary substantially within each class."
    )

with tab2:
    st.header("Pixel Structure and Intensity")

    digit = st.selectbox("Select class to inspect", range(10), key="structure")
    candidates = np.where(labels == digit)[0]
    image_number = st.slider(
        "Image index within class", 0, len(candidates) - 1, 0
    )
    img = images[candidates[image_number]]

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(img, cmap="gray", vmin=0, vmax=16)
        ax.set_title(f"Digit {digit}: 8×8 intensity array")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        fig.colorbar(im, ax=ax, label="Intensity (0–16)")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(img.ravel(), bins=np.arange(18) - 0.5,
                color="#4179a8", edgecolor="white")
        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("Number of pixels")
        ax.set_title("Intensity distribution")
        st.pyplot(fig)
        plt.close(fig)

    st.metric("Average intensity", f"{img.mean():.2f}")
    st.metric("Nonzero pixels", int(np.count_nonzero(img)))
    st.write("**Pixel intensity matrix:**")
    st.dataframe(img.astype(int), use_container_width=True)

    st.subheader("Mean Images — Comparing Two Classes")
    class_a = st.selectbox("First digit", range(10), index=3)
    class_b = st.selectbox("Second digit", range(10), index=8)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, cls in zip(axes, [class_a, class_b]):
        mean_img = images[labels == cls].mean(axis=0)
        ax.imshow(mean_img, cmap="viridis", vmin=0, vmax=16)
        ax.set_title(f"Mean image: {cls}")
        ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "These are grayscale images, so they have one intensity channel "
        "rather than separate RGB channels. Bright mean-image regions "
        "represent strokes that occur consistently across samples."
    )

with tab3:
    st.header("Classifier Behavior and Occlusion Saliency")
    st.metric("Held-out test accuracy", f"{accuracy:.1%}")

    st.write(
        "The random forest is trained on 75% of the images and evaluated "
        "on the remaining 25%. Occlusion saliency shows how covering "
        "small regions changes confidence in a selected class."
    )

    choice = st.slider("Select a held-out test image", 0, len(test_idx) - 1, 0)
    sample_id = test_idx[choice]
    img = images[sample_id]
    true_label = int(labels[sample_id])
    pred = int(model.predict([img.ravel()])[0])

    st.write(f"**True label:** {true_label} | **Prediction:** {pred}")

    patch = st.select_slider("Occlusion patch size", options=[1, 3, 5], value=3)
    target_class = st.radio(
        "Explain confidence for", ["Predicted class", "True class"],
        horizontal=True
    )
    cls = pred if target_class == "Predicted class" else true_label

    @st.cache_data(show_spinner=False)
    def calculate_saliency(sample_id, cls, patch):
        current_img = images[sample_id]
        base = model.predict_proba([current_img.ravel()])[0][cls]
        variants = []
        radius = patch // 2

        for i in range(8):
            for j in range(8):
                changed = current_img.copy()
                changed[
                    max(0, i-radius):min(8, i+radius+1),
                    max(0, j-radius):min(8, j+radius+1)
                ] = 0
                variants.append(changed.ravel())

        probabilities = model.predict_proba(np.asarray(variants))[:, cls]
        return (base - probabilities).reshape(8, 8), base

    saliency, baseline = calculate_saliency(sample_id, cls, patch)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=16)
    axes[0].set_title("Original image")

    axes[1].imshow(saliency, cmap="inferno")
    axes[1].set_title("Occlusion saliency")

    axes[2].imshow(img, cmap="gray", vmin=0, vmax=16)
    overlay = axes[2].imshow(
        np.maximum(saliency, 0), cmap="inferno",
        alpha=0.65, interpolation="nearest"
    )
    axes[2].set_title("Saliency overlay")

    for ax in axes:
        ax.axis("off")
    fig.colorbar(overlay, ax=axes, shrink=0.7,
                 label="Confidence decrease")
    st.pyplot(fig)
    plt.close(fig)

    st.write(f"Baseline confidence for class {cls}: **{baseline:.1%}**")
    st.info(
        "Brighter positive saliency values identify image regions whose "
        "removal decreases model confidence. Negative values mean that "
        "covering the region increases confidence. This explains model "
        "sensitivity, not necessarily human reasoning or causality."
    )

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(labels[test_idx], test_predictions,
                          labels=np.arange(10))
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title("Held-out test confusion matrix")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

with tab4:
    st.header("Interpretation and Limitations")

    st.markdown("""
    ### 1. Montage and Data Quality
    The montage reveals substantial variation in how people write the
    same digit. Some samples have faint or incomplete-looking strokes.
    Visual inspection can identify candidates for further review,
    but unusual handwriting is not automatically mislabeled.

    ### 2. Class Balance
    The digit classes are approximately balanced. This reduces the
    risk that raw accuracy is dominated by a single common class.

    ### 3. Structural Visualization
    Each image contains 64 grayscale intensity values ranging from
    0 to 16. Histograms show background versus stroke intensity.
    Mean images highlight consistent stroke locations while blurring
    individual writing styles.

    ### 4. Saliency and Model Behavior
    Occlusion saliency measures the change in predicted probability
    when a patch is hidden. Strong positive regions suggest pixels
    important to the classifier's decision. Results depend on patch
    size and the chosen class; occlusion can also create unnatural
    image patterns.

    ### 5. Limitations
    - Digits are only 8×8 pixels and are not representative of
      modern high-resolution image datasets.
    - Random forests do not expose convolutional feature maps.
    - Occlusion is a sensitivity analysis, not proof of causality.
    - Human inspection is still needed before declaring label errors.

    **Dataset:** scikit-learn Digits (1,797 handwritten digit images).
    **Method:** Stratified 75/25 train-test split; Random Forest;
    sliding-window occlusion.
    """)
