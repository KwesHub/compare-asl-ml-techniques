# ASL Hand Pose Recognition — Comparing ML Techniques

A machine learning pipeline that classifies American Sign Language (ASL) hand gestures from images. Built to compare multiple classification approaches, with a custom kNN implementation as the centrepiece.

## Results

| Model | Test Accuracy |
|---|---|
| **kNN (from scratch)** | **95.36%** |
| kNN (scikit-learn) | 94.80% |
| Random Forest | 93.12% |
| Decision Tree | 89.54% |

The from-scratch kNN outperformed all baseline classifiers, built using only Python standard libraries — no NumPy, no scikit-learn.

## Pipeline Overview

```
Raw Images → Feature Extraction → Preprocessing → Classification → Evaluation
```

1. **Feature Extraction** (`01_feature_extraction.py`) — Extracts 63 geometric hand landmark features per image using MediaPipe
2. **Preprocessing** (`02_data_preprocessing.py`) — Cleans data, applies SMOTE for class balancing, normalises features
3. **kNN from Scratch** (`03_knn_from_scratch.py`) — Full kNN implementation using only Python standard library (`math`, `collections`, `csv`)
4. **Supervised Learning** (`04_supervised_learning.py`) — Decision Tree, kNN, and Random Forest via scikit-learn for comparison
5. **Unsupervised Learning** (`05_unsupervised_learning.py`) — K-Means and Hierarchical clustering
6. **Visualisation** (`06_visualization.py`) — Confusion matrices, accuracy plots, clustering output

## Tech Stack

- **Python 3.9**
- MediaPipe — hand landmark detection
- OpenCV — image loading and processing
- scikit-learn — baseline classifiers and evaluation metrics
- imbalanced-learn — SMOTE oversampling
- pandas, matplotlib, seaborn — data handling and visualisation

## Setup

```bash
pip install opencv-python mediapipe "numpy<2" pandas scikit-learn matplotlib seaborn scipy imbalanced-learn
python main.py --all
```

**Note:** NumPy must be `< 2.0` for MediaPipe compatibility. Python 3.9 recommended.

## Key Design Decision

The kNN classifier (`03_knn_from_scratch.py`) uses **Euclidean distance** computed via `math.sqrt()` and **majority voting** via `collections.Counter` — zero external ML libraries. This was deliberately constrained to demonstrate understanding of the algorithm mechanics rather than library usage.
