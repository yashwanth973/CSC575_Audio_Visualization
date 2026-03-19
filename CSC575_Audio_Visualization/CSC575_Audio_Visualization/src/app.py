from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"

st.set_page_config(page_title="Audio Visualization Project", layout="wide")
st.title("CSC575 Audio Visualization and Retrieval")

st.write(
    "This app shows extracted audio features, PCA/t-SNE visualizations, "
    "and nearest-neighbor retrieval results."
)

features_path = OUTPUT_DIR / "audio_features.csv"
pca_path = OUTPUT_DIR / "pca_coordinates.csv"
tsne_path = OUTPUT_DIR / "tsne_coordinates.csv"
neighbors_path = OUTPUT_DIR / "top_neighbors_euclidean.csv"

missing = [p.name for p in [features_path, pca_path, neighbors_path] if not p.exists()]
if missing:
    st.error(
        "Missing required output files. Run `python src/run_pipeline.py` first.\n\n"
        + "\n".join(missing)
    )
    st.stop()

features_df = pd.read_csv(features_path)
pca_df = pd.read_csv(pca_path)
neighbors_df = pd.read_csv(neighbors_path)

has_tsne = tsne_path.exists()
if has_tsne:
    tsne_df = pd.read_csv(tsne_path)

tabs = ["Features", "PCA", "Neighbors"]
if has_tsne:
    tabs.insert(2, "t-SNE")

tab_objects = st.tabs(tabs)

tab_index = 0

with tab_objects[tab_index]:
    st.subheader("Extracted Feature Table")
    st.dataframe(features_df, use_container_width=True)
tab_index += 1

with tab_objects[tab_index]:
    st.subheader("PCA Visualization")
    fig = px.scatter(
        pca_df,
        x="pca_1",
        y="pca_2",
        text="file_name",
        title="PCA Map of Audio Files"
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)
tab_index += 1

if has_tsne:
    with tab_objects[tab_index]:
        st.subheader("t-SNE Visualization")
        fig = px.scatter(
            tsne_df,
            x="tsne_1",
            y="tsne_2",
            text="file_name",
            title="t-SNE Map of Audio Files"
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
    tab_index += 1

with tab_objects[tab_index]:
    st.subheader("Nearest Neighbor Retrieval")
    query_file = st.selectbox("Select a query file", neighbors_df["query_file"].tolist())

    row = neighbors_df[neighbors_df["query_file"] == query_file].iloc[0]
    st.write(f"### Query: {query_file}")

    results = []
    for i in range(1, 6):
        neighbor = row.get(f"neighbor_{i}")
        distance = row.get(f"distance_{i}")
        if pd.notna(neighbor):
            results.append({
                "Rank": i,
                "Neighbor File": neighbor,
                "Distance": distance
            })

    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.write("### Available audio files")
    audio_files = sorted(AUDIO_DIR.glob("*"))
    audio_lookup = {p.stem.replace("_", " ").strip()[:35] + ("..." if len(p.stem.replace("_", " ").strip()) > 35 else ""): p for p in audio_files}

    for label, path in audio_lookup.items():
        if label == query_file:
            st.write(f"**Query audio: {path.name}**")
            st.audio(str(path))

    for item in results:
        for label, path in audio_lookup.items():
            if label == item["Neighbor File"]:
                st.write(f"**Rank {item['Rank']}: {path.name}**")
                st.audio(str(path))
                break