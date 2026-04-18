<<<<<<< HEAD
import warnings
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_audio_files(audio_dir: Path):
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    return sorted([p for p in audio_dir.rglob("*") if p.suffix.lower() in exts])


def make_clean_label(file_path: Path, max_len: int = 35) -> str:
    name = file_path.stem.replace("_", " ").strip()
    if len(name) > max_len:
        return name[:max_len] + "..."
    return name


def extract_features(file_path: Path, sr: int = 22050, duration: int = 10):
    """
    Extract a compact feature vector from an audio file.
    """
    y, sr = librosa.load(file_path, sr=sr, mono=True, duration=duration)

    if y is None or len(y) == 0:
        raise ValueError(f"Could not load audio: {file_path.name}")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    features = {}

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


def save_waveform_plot(y, sr, file_stem: str, title_label: str):
    out_path = OUTPUT_DIR / f"{file_stem}_waveform.png"
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform - {title_label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_spectrogram_plot(y, sr, file_stem: str, title_label: str):
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


def save_fft_plot(y, sr, file_stem: str, title_label: str):
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


def build_neighbor_table(file_names, distance_matrix, metric_name="euclidean", top_k=5):
    rows = []
    for i, query_name in enumerate(file_names):
        distances = distance_matrix[i].copy()
        order = np.argsort(distances)
        neighbors = [idx for idx in order if idx != i][:top_k]

        row = {"query_file": query_name, "metric": metric_name}
        for rank, idx in enumerate(neighbors, start=1):
            row[f"neighbor_{rank}"] = file_names[idx]
            row[f"distance_{rank}"] = float(distances[idx])
        rows.append(row)

    return pd.DataFrame(rows)


def save_embedding_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path):
    plt.figure(figsize=(10, 7))
    plt.scatter(df[x_col], df[y_col], alpha=0.85)

    for _, row in df.iterrows():
        plt.annotate(
            row["file_name"],
            (row[x_col], row[y_col]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {AUDIO_DIR}")

    audio_files = collect_audio_files(AUDIO_DIR)

    if len(audio_files) == 0:
        raise FileNotFoundError("No audio files found. Add files to data/audio/")

    print(f"Found {len(audio_files)} audio files")

    rows = []
    saved_example_outputs = False

    for file_path in audio_files:
        print(f"Processing: {file_path.name}")
        try:
            y, sr, features = extract_features(file_path)
            clean_name = make_clean_label(file_path)

            row = {
                "file_name": clean_name,
                "original_file_name": file_path.name,
                "file_path": str(file_path),
            }
            row.update(features)
            rows.append(row)

            if not saved_example_outputs:
                stem = file_path.stem
                save_waveform_plot(y, sr, stem, clean_name)
                save_spectrogram_plot(y, sr, stem, clean_name)
                save_fft_plot(y, sr, stem, clean_name)
                saved_example_outputs = True

        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")

    if len(rows) < 2:
        raise ValueError("Need at least 2 valid audio files to continue.")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "audio_features.csv", index=False)

    feature_cols = [
        c for c in df.columns
        if c not in ["file_name", "original_file_name", "file_path"]
    ]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Distance matrices
    euclidean_dist = pairwise_distances(X_scaled, metric="euclidean")
    cosine_dist = pairwise_distances(X_scaled, metric="cosine")

    pd.DataFrame(
        euclidean_dist,
        index=df["file_name"],
        columns=df["file_name"]
    ).to_csv(OUTPUT_DIR / "distance_matrix_euclidean.csv")

    pd.DataFrame(
        cosine_dist,
        index=df["file_name"],
        columns=df["file_name"]
    ).to_csv(OUTPUT_DIR / "distance_matrix_cosine.csv")

    euclidean_neighbors = build_neighbor_table(df["file_name"].tolist(), euclidean_dist, "euclidean")
    cosine_neighbors = build_neighbor_table(df["file_name"].tolist(), cosine_dist, "cosine")

    euclidean_neighbors.to_csv(OUTPUT_DIR / "top_neighbors_euclidean.csv", index=False)
    cosine_neighbors.to_csv(OUTPUT_DIR / "top_neighbors_cosine.csv", index=False)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "file_name": df["file_name"],
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

    # t-SNE only if enough files are available
    if len(df) >= 3:
        perplexity = min(5, len(df) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_coords = tsne.fit_transform(X_scaled)

        tsne_df = pd.DataFrame({
            "file_name": df["file_name"],
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
        print("Saved t-SNE outputs.")
    else:
        print("Skipped t-SNE: need at least 3 audio files.")

    print("\nDone. Outputs saved in the outputs folder:")
    print("- audio_features.csv")
    print("- distance_matrix_euclidean.csv")
    print("- distance_matrix_cosine.csv")
    print("- top_neighbors_euclidean.csv")
    print("- top_neighbors_cosine.csv")
    print("- pca_coordinates.csv")
    print("- pca_map.png")
    print("- sample waveform / spectrogram / fft PNGs")
    if len(df) >= 3:
        print("- tsne_coordinates.csv")
        print("- tsne_map.png")


if __name__ == "__main__":
=======
import warnings
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_audio_files(audio_dir: Path):
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    return sorted([p for p in audio_dir.rglob("*") if p.suffix.lower() in exts])


def make_clean_label(file_path: Path, max_len: int = 35) -> str:
    name = file_path.stem.replace("_", " ").strip()
    if len(name) > max_len:
        return name[:max_len] + "..."
    return name


def extract_features(file_path: Path, sr: int = 22050, duration: int = 10):
    """
    Extract a compact feature vector from an audio file.
    """
    y, sr = librosa.load(file_path, sr=sr, mono=True, duration=duration)

    if y is None or len(y) == 0:
        raise ValueError(f"Could not load audio: {file_path.name}")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    features = {}

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


def save_waveform_plot(y, sr, file_stem: str, title_label: str):
    out_path = OUTPUT_DIR / f"{file_stem}_waveform.png"
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform - {title_label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_spectrogram_plot(y, sr, file_stem: str, title_label: str):
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


def save_fft_plot(y, sr, file_stem: str, title_label: str):
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


def build_neighbor_table(file_names, distance_matrix, metric_name="euclidean", top_k=5):
    rows = []
    for i, query_name in enumerate(file_names):
        distances = distance_matrix[i].copy()
        order = np.argsort(distances)
        neighbors = [idx for idx in order if idx != i][:top_k]

        row = {"query_file": query_name, "metric": metric_name}
        for rank, idx in enumerate(neighbors, start=1):
            row[f"neighbor_{rank}"] = file_names[idx]
            row[f"distance_{rank}"] = float(distances[idx])
        rows.append(row)

    return pd.DataFrame(rows)


def save_embedding_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path):
    plt.figure(figsize=(10, 7))
    plt.scatter(df[x_col], df[y_col], alpha=0.85)

    for _, row in df.iterrows():
        plt.annotate(
            row["file_name"],
            (row[x_col], row[y_col]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {AUDIO_DIR}")

    audio_files = collect_audio_files(AUDIO_DIR)

    if len(audio_files) == 0:
        raise FileNotFoundError("No audio files found. Add files to data/audio/")

    print(f"Found {len(audio_files)} audio files")

    rows = []
    saved_example_outputs = False

    for file_path in audio_files:
        print(f"Processing: {file_path.name}")
        try:
            y, sr, features = extract_features(file_path)
            clean_name = make_clean_label(file_path)

            row = {
                "file_name": clean_name,
                "original_file_name": file_path.name,
                "file_path": str(file_path),
            }
            row.update(features)
            rows.append(row)

            if not saved_example_outputs:
                stem = file_path.stem
                save_waveform_plot(y, sr, stem, clean_name)
                save_spectrogram_plot(y, sr, stem, clean_name)
                save_fft_plot(y, sr, stem, clean_name)
                saved_example_outputs = True

        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")

    if len(rows) < 2:
        raise ValueError("Need at least 2 valid audio files to continue.")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "audio_features.csv", index=False)

    feature_cols = [
        c for c in df.columns
        if c not in ["file_name", "original_file_name", "file_path"]
    ]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Distance matrices
    euclidean_dist = pairwise_distances(X_scaled, metric="euclidean")
    cosine_dist = pairwise_distances(X_scaled, metric="cosine")

    pd.DataFrame(
        euclidean_dist,
        index=df["file_name"],
        columns=df["file_name"]
    ).to_csv(OUTPUT_DIR / "distance_matrix_euclidean.csv")

    pd.DataFrame(
        cosine_dist,
        index=df["file_name"],
        columns=df["file_name"]
    ).to_csv(OUTPUT_DIR / "distance_matrix_cosine.csv")

    euclidean_neighbors = build_neighbor_table(df["file_name"].tolist(), euclidean_dist, "euclidean")
    cosine_neighbors = build_neighbor_table(df["file_name"].tolist(), cosine_dist, "cosine")

    euclidean_neighbors.to_csv(OUTPUT_DIR / "top_neighbors_euclidean.csv", index=False)
    cosine_neighbors.to_csv(OUTPUT_DIR / "top_neighbors_cosine.csv", index=False)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "file_name": df["file_name"],
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

    # t-SNE only if enough files are available
    if len(df) >= 3:
        perplexity = min(5, len(df) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_coords = tsne.fit_transform(X_scaled)

        tsne_df = pd.DataFrame({
            "file_name": df["file_name"],
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
        print("Saved t-SNE outputs.")
    else:
        print("Skipped t-SNE: need at least 3 audio files.")

    print("\nDone. Outputs saved in the outputs folder:")
    print("- audio_features.csv")
    print("- distance_matrix_euclidean.csv")
    print("- distance_matrix_cosine.csv")
    print("- top_neighbors_euclidean.csv")
    print("- top_neighbors_cosine.csv")
    print("- pca_coordinates.csv")
    print("- pca_map.png")
    print("- sample waveform / spectrogram / fft PNGs")
    if len(df) >= 3:
        print("- tsne_coordinates.csv")
        print("- tsne_map.png")


if __name__ == "__main__":
>>>>>>> 7f47d451c7a5e4dd870316c6ef400bc7644df07c
    main()