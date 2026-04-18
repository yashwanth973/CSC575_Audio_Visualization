from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"

st.set_page_config(page_title="Audio Visualization and Retrieval", layout="wide")
st.title("Interactive Visualization and Similarity-Based Exploration of Audio Collections")

st.write(
    "This interface lets you inspect the dataset, browse audio features, "
    "explore PCA / t-SNE maps, review retrieval quality, and search neighbors."
)

required_files = [
    OUTPUT_DIR / "audio_features.csv",
    OUTPUT_DIR / "dataset_summary.csv",
    OUTPUT_DIR / "top_neighbors_euclidean.csv",
    OUTPUT_DIR / "top_neighbors_cosine.csv",
    OUTPUT_DIR / "pca_coordinates.csv",
    OUTPUT_DIR / "evaluation_metrics.csv",
    OUTPUT_DIR / "performance_indicators.csv",
]

missing = [p.name for p in required_files if not p.exists()]
if missing:
    st.error("Missing required outputs. Run `python src/run_pipeline.py` first.\n\n" + "\n".join(missing))
    st.stop()

features_df = pd.read_csv(OUTPUT_DIR / "audio_features.csv")
dataset_df = pd.read_csv(OUTPUT_DIR / "dataset_summary.csv")
pca_df = pd.read_csv(OUTPUT_DIR / "pca_coordinates.csv")
euclidean_df = pd.read_csv(OUTPUT_DIR / "top_neighbors_euclidean.csv")
cosine_df = pd.read_csv(OUTPUT_DIR / "top_neighbors_cosine.csv")
eval_df = pd.read_csv(OUTPUT_DIR / "evaluation_metrics.csv")
perf_df = pd.read_csv(OUTPUT_DIR / "performance_indicators.csv")

has_tsne = (OUTPUT_DIR / "tsne_coordinates.csv").exists()
if has_tsne:
    tsne_df = pd.read_csv(OUTPUT_DIR / "tsne_coordinates.csv")

tab_names = ["Overview", "Performance", "Features", "PCA", "Retrieval"]
if has_tsne:
    tab_names.insert(4, "t-SNE")

tabs = st.tabs(tab_names)

# ---------------- Overview ----------------
with tabs[0]:
    st.subheader("Dataset Overview")
    st.dataframe(dataset_df, use_container_width=True)

    st.subheader("Evaluation Metrics")
    st.dataframe(eval_df, use_container_width=True)

# ---------------- Performance ----------------
with tabs[1]:
    st.subheader("Performance Indicators")
    st.dataframe(perf_df, use_container_width=True)

# ---------------- Features ----------------
with tabs[2]:
    st.subheader("Feature Table")

    labels = sorted(features_df["label"].dropna().unique().tolist())
    selected_labels = st.multiselect("Filter by label", labels, default=labels)

    search_text = st.text_input("Search by file name", value="").strip().lower()

    filtered = features_df.copy()
    if selected_labels:
        filtered = filtered[filtered["label"].isin(selected_labels)]
    if search_text:
        filtered = filtered[filtered["file_name"].str.lower().str.contains(search_text)]

    st.dataframe(filtered, use_container_width=True)

# ---------------- PCA ----------------
with tabs[3]:
    st.subheader("PCA Map")

    labels = sorted(pca_df["label"].dropna().unique().tolist())
    selected_labels = st.multiselect("Choose labels for PCA", labels, default=labels, key="pca_labels")

    pca_filtered = pca_df[pca_df["label"].isin(selected_labels)] if selected_labels else pca_df

    fig = px.scatter(
        pca_filtered,
        x="pca_1",
        y="pca_2",
        color="label",
        hover_name="file_name",
        title="PCA Projection of Audio Files"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- t-SNE ----------------
retrieval_tab_index = 4
if has_tsne:
    with tabs[4]:
        st.subheader("t-SNE Map")

        labels = sorted(tsne_df["label"].dropna().unique().tolist())
        selected_labels = st.multiselect("Choose labels for t-SNE", labels, default=labels, key="tsne_labels")

        tsne_filtered = tsne_df[tsne_df["label"].isin(selected_labels)] if selected_labels else tsne_df

        fig = px.scatter(
            tsne_filtered,
            x="tsne_1",
            y="tsne_2",
            color="label",
            hover_name="file_name",
            title="t-SNE Projection of Audio Files"
        )
        st.plotly_chart(fig, use_container_width=True)

    retrieval_tab_index = 5

# ---------------- Retrieval ----------------
with tabs[retrieval_tab_index]:
    st.subheader("Similarity Search")

    metric = st.selectbox("Similarity metric", ["euclidean", "cosine"])
    top_k = st.slider("Top K neighbors", min_value=1, max_value=5, value=5)

    neighbor_df = euclidean_df if metric == "euclidean" else cosine_df

    all_labels = sorted(neighbor_df["query_label"].dropna().unique().tolist())
    selected_label = st.selectbox("Filter query by label", ["All"] + all_labels)

    filtered_queries = neighbor_df.copy()
    if selected_label != "All":
        filtered_queries = filtered_queries[filtered_queries["query_label"] == selected_label]

    query_file = st.selectbox("Choose query file", filtered_queries["query_file"].tolist())

    row = neighbor_df[neighbor_df["query_file"] == query_file].iloc[0]

    results = []
    for i in range(1, top_k + 1):
        neighbor = row.get(f"neighbor_{i}")
        neighbor_label = row.get(f"neighbor_{i}_label")
        distance = row.get(f"distance_{i}")
        if pd.notna(neighbor):
            results.append({
                "Rank": i,
                "Neighbor File": neighbor,
                "Neighbor Label": neighbor_label,
                "Distance": distance,
            })

    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.subheader("Audio Playback")

    audio_paths = {}
    for path in AUDIO_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            clean_label = path.stem.replace("_", " ").strip()
            if len(clean_label) > 40:
                clean_label = clean_label[:40] + "..."
            audio_paths[clean_label] = path

    if query_file in audio_paths:
        st.write(f"**Query Audio:** {audio_paths[query_file].name}")
        st.audio(str(audio_paths[query_file]))

    for item in results:
        neighbor_name = item["Neighbor File"]
        if neighbor_name in audio_paths:
            st.write(f"**Rank {item['Rank']} — {audio_paths[neighbor_name].name}**")
            st.audio(str(audio_paths[neighbor_name]))