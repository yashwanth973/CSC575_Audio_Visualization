<<<<<<< HEAD
<<<<<<< HEAD
# CSC575\_Audio\_Visualization

This repository implements a content-based music retrieval and visualization system, focusing on the DP3 - Content-Based Visualization and Browsing of Sound Effects project. The core objective is to move beyond metadata-dependent searching by organizing large audio collections (2,000–5,000 clips) based on their intrinsic acoustic properties.

# 17/04/2026
# CSC575_Audio_Visualization

Interactive Visualization and Similarity-Based Exploration of Audio Collections.

## Features
- Audio preprocessing
- MIR feature extraction:
  - MFCC
  - Chroma
  - Spectral centroid
  - Spectral bandwidth
  - Spectral rolloff
  - Zero-crossing rate
  - RMS energy
- Similarity computation:
  - Euclidean distance
  - Cosine distance
- Visualizations:
  - Waveform
  - Spectrogram
  - FFT magnitude spectrum
  - PCA map
  - t-SNE map
- Retrieval outputs:
  - Top nearest neighbors
  - Retrieval summary
  - Evaluation metrics (Precision@K, Recall@K, MRR)
- Streamlit browser with search and filters

## Dataset format
Place audio files inside subfolders to represent labels/classes:

```text
data/audio/
├── class1/
│   ├── file1.wav
│   └── file2.wav
├── class2/
│   ├── file3.wav
│   └── file4.wav

## Project structure

CSC575_Audio_Visualization/
│
├── data/
│   └── audio/
├── outputs/
├── src/
│   ├── utils.py
│   ├── run_pipeline.py
│   └── app.py
├── requirements.txt
├── README.md
└── .gitignore

=======
>>>>>>> 7f47d451c7a5e4dd870316c6ef400bc7644df07c
=======
>>>>>>> 7f47d451c7a5e4dd870316c6ef400bc7644df07c
