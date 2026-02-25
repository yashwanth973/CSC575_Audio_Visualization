# CSC575_Audio_Visualization
This repository implements a content-based music retrieval and visualization system, focusing on the DP3 - Content-Based Visualization and Browsing of Sound Effects project. The core objective is to move beyond metadata-dependent searching by organizing large audio collections (2,000–5,000 clips) based on their intrinsic acoustic properties.
## Yashwanthkrishna Nagharaj
Objective 1 (Engineer a robust feature extraction and similarity engine)

PI1 (basic): Implement a batch-processing script using Librosa to extract MFCCs and spectral centroids for a 2,000-clip subset.

PI2 (basic): Generate and store a pairwise distance matrix using standard Euclidean distance.

PI3 (expected): Expand the feature set to include Chroma and Zero-Crossing Rate to better capture harmonic and percussive variance.

PI4 (expected): Compare Euclidean vs. Cosine similarity metrics to determine which yields more perceptually accurate neighbors.

PI5 (advanced): Implement an optimized indexing structure (e.g., BallTree or FAISS) to support similarity queries on the distance matrix in under 100ms.

Objective 2 (Quantitative system benchmarking and evaluation)

PI1 (basic): Document and log execution times for the initial extraction pipeline.

PI2 (basic): Manually verify similarity for a random sample of 20 query points and their top 5 neighbors.

PI3 (expected): Calculate the Silhouette score for various 2D projections to mathematically quantify cluster quality.

PI4 (expected): Perform "Precision-at-k" checks using existing metadata tags as a ground-truth baseline to validate the clustering.

PI5 (advanced): Lead the technical drafting and LaTeX formatting of the final ISMIR-style paper, ensuring rigorous methodology documentation.

## Arunkumar Krishnamoorthy
Objective 1 (Implement robust dimensionality reduction for 2D visualization)

PI1 (basic): Successfully project high-dimensional feature vectors into 2D using PCA for a linear baseline.

PI2 (basic): Establish a baseline visualization of the audio collection using a standard Plotly scatter plot.

PI3 (expected): Implement non-linear projections using t-SNE and UMAP for superior visual cluster separation.

PI4 (expected): Develop a hyperparameter tuning script to optimize UMAP stability (n_neighbors, min_dist) across the dataset.

PI5 (advanced): Create a "dynamic projection" toggle in the UI allowing users to switch between different dimensionality reduction methods in real-time.

Objective 2 (Develop a responsive and interactive browsing interface)

PI1 (basic): Create a basic web-based UI (Dash/Streamlit) that displays the 2D audio map.

PI2 (basic): Implement a "click-to-play" feature for immediate audio feedback when a node is selected.

PI3 (expected): Build a "Nearest Neighbor" visualizer that highlights related nodes upon selection.

PI4 (expected): Integrate metadata-based color coding to visually validate if clusters align with known genre or instrument tags.

PI5 (advanced): Implement real-time audio playback with a waveform preview or spectrogram overlay for selected sounds.
