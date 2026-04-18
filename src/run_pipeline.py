from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from utils import collect_audio_records, make_clean_label

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose: "handcrafted" or "openl3"
# openl3 is optional and not required for the project to run.
EMBEDDING_MODE = "handcrafted"


def try_openl3_embedding(y: np.ndarray, sr: int) -> np.ndarray | None:
    """
    Optional stronger embedding path.
    Returns None if openl3 is unavailable.
    """
    try:
        import openl3  # type: ignore

        emb, _ = openl3.get_audio_embedding(
            y,
            sr,
            content_type="music",
            input_repr="mel256",
            embedding_size=512,
        )
        return np.mean(emb, axis=0)
    except Exception:
        return None


def extract_handcrafted_features(file_path: Path, sr: int = 22050, duration: int = 10) -> Tuple[np.ndarray, int, Dict[str, float]]:
    y, sr = librosa.load(file_path, sr=sr, mono=True, duration=duration)

    if y is None or len(y) == 0:
        raise ValueError(f"Could not load audio: {file_path.name}")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    features: Dict[str, float] = {}

    for i in range(mfcc.shape[0]):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    for i in range(chroma.shape[0]):
        features[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))

    features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
    features["spectral_centroid_std"] = float(np.std(spectral_centroid))
    features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))
    features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
    features["spectral_rolloff_std"] = float(np.std(spectral_rolloff))
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))
    features["duration_seconds"] = float(librosa.get_duration(y=y, sr=sr))

    return y, sr, features


def extract_features(file_path: Path, sr: int = 22050, duration: int = 10) -> Tuple[np.ndarray, int, Dict[str, float]]:
    y, sr, features = extract_handcrafted_features(file_path, sr=sr, duration=duration)

    if EMBEDDING_MODE == "openl3":
        openl3_vec = try_openl3_embedding(y, sr)
        if openl3_vec is not None:
            for i, val in enumerate(openl3_vec):
                features[f"openl3_{i+1}"] = float(val)
        else:
            print(f"OpenL3 unavailable for {file_path.name}; falling back to handcrafted features.")

    return y, sr, features


def save_waveform_plot(y: np.ndarray, sr: int, file_stem: str, title_label: str) -> None:
    out_path = OUTPUT_DIR / f"{file_stem}_waveform.png"
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform - {title_label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_spectrogram_plot(y: np.ndarray, sr: int, file_stem: str, title_label: str) -> None:
    out_path = OUTPUT_DIR / f"{file_stem}_spectrogram.png"
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram - {title_label}")
    plt.xlabel("Time")
    plt.ylabel("Hz")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_fft_plot(y: np.ndarray, sr: int, file_stem: str, title_label: str) -> None:
    out_path = OUTPUT_DIR / f"{file_stem}_fft.png"
    n = len(y)
    fft_vals = np.fft.rfft(y)
    fft_freq = np.fft.rfftfreq(n, d=1 / sr)
    magnitude = np.abs(fft_vals)

    plt.figure(figsize=(12, 4))
    plt.plot(fft_freq, magnitude)
    plt.title(f"FFT Magnitude Spectrum - {title_label}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_embedding_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 7))

    if "label" in df.columns:
        labels = sorted(df["label"].unique())
        for label in labels:
            sub = df[df["label"] == label]
            plt.scatter(sub[x_col], sub[y_col], alpha=0.85, label=label)
        if len(labels) <= 10:
            plt.legend(fontsize=8)
    else:
        plt.scatter(df[x_col], df[y_col], alpha=0.85)

    for _, row in df.iterrows():
        plt.annotate(
            row["file_name"],
            (row[x_col], row[y_col]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points"
        )

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_neighbor_table(file_names: List[str], labels: List[str], distance_matrix: np.ndarray, metric_name: str, top_k: int = 5) -> pd.DataFrame:
    rows = []

    for i, query_name in enumerate(file_names):
        distances = distance_matrix[i].copy()
        order = np.argsort(distances)
        neighbors = [idx for idx in order if idx != i][:top_k]

        row = {
            "query_file": query_name,
            "query_label": labels[i],
            "metric": metric_name,
        }

        for rank, idx in enumerate(neighbors, start=1):
            row[f"neighbor_{rank}"] = file_names[idx]
            row[f"neighbor_{rank}_label"] = labels[idx]
            row[f"distance_{rank}"] = float(distances[idx])

        rows.append(row)

    return pd.DataFrame(rows)


def create_retrieval_summary(euclidean_neighbors: pd.DataFrame, cosine_neighbors: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "query_file": euclidean_neighbors["query_file"],
        "query_label": euclidean_neighbors["query_label"],
        "euclidean_top1": euclidean_neighbors["neighbor_1"],
        "euclidean_top1_label": euclidean_neighbors["neighbor_1_label"],
        "cosine_top1": cosine_neighbors["neighbor_1"],
        "cosine_top1_label": cosine_neighbors["neighbor_1_label"],
    })
    summary["same_top1_neighbor"] = summary["euclidean_top1"] == summary["cosine_top1"]
    summary["euclidean_top1_correct_label"] = summary["query_label"] == summary["euclidean_top1_label"]
    summary["cosine_top1_correct_label"] = summary["query_label"] == summary["cosine_top1_label"]
    summary.to_csv(OUTPUT_DIR / "retrieval_summary.csv", index=False)
    return summary


def precision_at_k(neighbor_df: pd.DataFrame, k: int) -> float:
    scores = []
    for _, row in neighbor_df.iterrows():
        query_label = row["query_label"]
        correct = 0
        for rank in range(1, k + 1):
            if row.get(f"neighbor_{rank}_label") == query_label:
                correct += 1
        scores.append(correct / k)
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(neighbor_df: pd.DataFrame, k: int, label_counts: Dict[str, int]) -> float:
    recalls = []
    for _, row in neighbor_df.iterrows():
        query_label = row["query_label"]
        total_relevant = max(label_counts.get(query_label, 1) - 1, 1)
        correct = 0
        for rank in range(1, k + 1):
            if row.get(f"neighbor_{rank}_label") == query_label:
                correct += 1
        recalls.append(correct / total_relevant)
    return float(np.mean(recalls)) if recalls else 0.0


def mean_reciprocal_rank(neighbor_df: pd.DataFrame, top_k: int = 5) -> float:
    rr_scores = []
    for _, row in neighbor_df.iterrows():
        query_label = row["query_label"]
        rr = 0.0
        for rank in range(1, top_k + 1):
            if row.get(f"neighbor_{rank}_label") == query_label:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)
    return float(np.mean(rr_scores)) if rr_scores else 0.0


def create_evaluation_metrics(euclidean_neighbors: pd.DataFrame, cosine_neighbors: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
    label_counts = pd.Series(labels).value_counts().to_dict()

    # evaluation is meaningful only if there is at least one repeated label
    repeated_labels = any(count >= 2 for count in label_counts.values())

    if not repeated_labels:
        eval_df = pd.DataFrame([
            ["Euclidean", "Precision@1", np.nan],
            ["Euclidean", "Precision@3", np.nan],
            ["Euclidean", "Precision@5", np.nan],
            ["Euclidean", "Recall@5", np.nan],
            ["Euclidean", "MRR", np.nan],
            ["Cosine", "Precision@1", np.nan],
            ["Cosine", "Precision@3", np.nan],
            ["Cosine", "Precision@5", np.nan],
            ["Cosine", "Recall@5", np.nan],
            ["Cosine", "MRR", np.nan],
        ], columns=["metric_type", "evaluation_metric", "value"])
        eval_df.to_csv(OUTPUT_DIR / "evaluation_metrics.csv", index=False)
        return eval_df

    rows = []
    for metric_name, neighbor_df in [("Euclidean", euclidean_neighbors), ("Cosine", cosine_neighbors)]:
        rows.append([metric_name, "Precision@1", precision_at_k(neighbor_df, 1)])
        rows.append([metric_name, "Precision@3", precision_at_k(neighbor_df, min(3, 5))])
        rows.append([metric_name, "Precision@5", precision_at_k(neighbor_df, 5)])
        rows.append([metric_name, "Recall@5", recall_at_k(neighbor_df, 5, label_counts)])
        rows.append([metric_name, "MRR", mean_reciprocal_rank(neighbor_df, 5)])

    eval_df = pd.DataFrame(rows, columns=["metric_type", "evaluation_metric", "value"])
    eval_df.to_csv(OUTPUT_DIR / "evaluation_metrics.csv", index=False)
    return eval_df


def create_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("label")
        .agg(
            num_files=("file_name", "count"),
            avg_duration_seconds=("duration_seconds", "mean"),
        )
        .reset_index()
        .sort_values("num_files", ascending=False)
    )
    summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)
    return summary


def create_performance_indicator_table(num_files: int, has_tsne: bool, has_repeated_labels: bool) -> pd.DataFrame:
    table = pd.DataFrame([
        ["Audio preprocessing", "Complete", "Pipeline loads and preprocesses audio files"],
        ["Feature extraction", "Complete", "MFCC, chroma, spectral, ZCR, RMS extracted"],
        ["Waveform visualization", "Complete", "Waveform PNG generated"],
        ["Spectrogram visualization", "Complete", "Spectrogram PNG generated"],
        ["FFT visualization", "Complete", "FFT magnitude plot generated"],
        ["Similarity computation", "Complete", "Euclidean and cosine matrices generated"],
        ["Nearest-neighbor retrieval", "Complete", "Neighbor CSV files generated"],
        ["PCA visualization", "Complete", "PCA coordinates and plot generated"],
        ["t-SNE visualization", "Complete" if has_tsne else "Partial", "Generated if at least 3 files are available"],
        ["Quantitative retrieval evaluation", "Complete" if has_repeated_labels else "Partial", "Requires repeated class labels"],
        ["Interactive browser", "Complete", "Available through Streamlit app"],
        ["Dataset size", f"{num_files} files", "More files improve evaluation strength"],
    ], columns=["Objective", "Status", "Evidence"])

    table.to_csv(OUTPUT_DIR / "performance_indicators.csv", index=False)
    return table


def main() -> None:
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {AUDIO_DIR}")

    audio_records = collect_audio_records(AUDIO_DIR)

    if len(audio_records) == 0:
        raise FileNotFoundError("No audio files found. Add files to data/audio/")

    print(f"Found {len(audio_records)} audio files")

    rows: List[Dict[str, object]] = []
    saved_example_outputs = False

    for record in audio_records:
        file_path = Path(record["file_path"])
        print(f"Processing: {file_path.name}")

        try:
            y, sr, features = extract_features(file_path)
            clean_name = make_clean_label(record["file_stem"])

            row: Dict[str, object] = {
                "file_name": clean_name,
                "original_file_name": record["original_file_name"],
                "file_path": record["file_path"],
                "label": record["label"],
            }
            row.update(features)
            rows.append(row)

            if not saved_example_outputs:
                save_waveform_plot(y, sr, file_path.stem, clean_name)
                save_spectrogram_plot(y, sr, file_path.stem, clean_name)
                save_fft_plot(y, sr, file_path.stem, clean_name)
                saved_example_outputs = True

        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")

    if len(rows) < 2:
        raise ValueError("Need at least 2 valid audio files to continue.")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "audio_features.csv", index=False)

    create_dataset_summary(df)

    feature_cols = [
        c for c in df.columns
        if c not in ["file_name", "original_file_name", "file_path", "label"]
    ]
    X = df[feature_cols].values
    labels = df["label"].tolist()
    file_names = df["file_name"].tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    euclidean_dist = pairwise_distances(X_scaled, metric="euclidean")
    cosine_dist = pairwise_distances(X_scaled, metric="cosine")

    pd.DataFrame(euclidean_dist, index=file_names, columns=file_names).to_csv(
        OUTPUT_DIR / "distance_matrix_euclidean.csv"
    )
    pd.DataFrame(cosine_dist, index=file_names, columns=file_names).to_csv(
        OUTPUT_DIR / "distance_matrix_cosine.csv"
    )

    euclidean_neighbors = build_neighbor_table(file_names, labels, euclidean_dist, "euclidean", top_k=5)
    cosine_neighbors = build_neighbor_table(file_names, labels, cosine_dist, "cosine", top_k=5)

    euclidean_neighbors.to_csv(OUTPUT_DIR / "top_neighbors_euclidean.csv", index=False)
    cosine_neighbors.to_csv(OUTPUT_DIR / "top_neighbors_cosine.csv", index=False)

    create_retrieval_summary(euclidean_neighbors, cosine_neighbors)

    label_counts = pd.Series(labels).value_counts().to_dict()
    has_repeated_labels = any(count >= 2 for count in label_counts.values())
    create_evaluation_metrics(euclidean_neighbors, cosine_neighbors, labels)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({
        "file_name": file_names,
        "label": labels,
        "pca_1": pca_coords[:, 0],
        "pca_2": pca_coords[:, 1],
    })
    pca_df.to_csv(OUTPUT_DIR / "pca_coordinates.csv", index=False)
    save_embedding_plot(
        pca_df,
        "pca_1",
        "pca_2",
        "PCA Projection of Audio Files",
        OUTPUT_DIR / "pca_map.png"
    )

    # t-SNE
    has_tsne = False
    if len(df) >= 3:
        perplexity = min(5, len(df) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_coords = tsne.fit_transform(X_scaled)

        tsne_df = pd.DataFrame({
            "file_name": file_names,
            "label": labels,
            "tsne_1": tsne_coords[:, 0],
            "tsne_2": tsne_coords[:, 1],
        })
        tsne_df.to_csv(OUTPUT_DIR / "tsne_coordinates.csv", index=False)
        save_embedding_plot(
            tsne_df,
            "tsne_1",
            "tsne_2",
            "t-SNE Projection of Audio Files",
            OUTPUT_DIR / "tsne_map.png"
        )
        has_tsne = True
        print("Saved t-SNE outputs.")
    else:
        print("Skipped t-SNE: need at least 3 audio files.")

    create_performance_indicator_table(len(df), has_tsne, has_repeated_labels)

    print("\nDone. Outputs saved in the outputs folder:")
    print("- audio_features.csv")
    print("- dataset_summary.csv")
    print("- distance_matrix_euclidean.csv")
    print("- distance_matrix_cosine.csv")
    print("- top_neighbors_euclidean.csv")
    print("- top_neighbors_cosine.csv")
    print("- retrieval_summary.csv")
    print("- evaluation_metrics.csv")
    print("- performance_indicators.csv")
    print("- pca_coordinates.csv")
    print("- pca_map.png")
    print("- sample waveform / spectrogram / fft PNGs")
    if has_tsne:
        print("- tsne_coordinates.csv")
        print("- tsne_map.png")


if __name__ == "__main__":
    main()