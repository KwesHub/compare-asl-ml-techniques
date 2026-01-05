"""
================================================================================
05_unsupervised_learning.py - Unsupervised Learning (Clustering)
================================================================================

Module:     CMP-6058A/7058A Artificial Intelligence
University: University of East Anglia (UEA)
Author:     [Group A108]

COURSEWORK PART 2d: Unsupervised Learning - Clustering Data

This script applies clustering algorithms to the ASL hand pose data:
    1. K-Means clustering
    2. Hierarchical (Agglomerative) clustering

WHAT IS UNSUPERVISED LEARNING?
    Unlike supervised learning, we don't use the class labels!
    Instead, we let the algorithm find natural groupings (clusters)
    in the data based on similarity.

WHAT WE DO:
    1. Remove class labels from the data
    2. Apply K-Means clustering
    3. Apply Hierarchical clustering
    4. Compare cluster assignments with actual labels
    5. Analyse how well clustering captures the true classes

LEARNING RESOURCES:
    - K-Means: https://www.geeksforgeeks.org/k-means-clustering-introduction/
    - Hierarchical: https://www.geeksforgeeks.org/hierarchical-clustering/
    - Silhouette Score: https://scikit-learn.org/stable/modules/clustering.html

================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

# Use non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Clustering algorithms
from sklearn.cluster import KMeans, AgglomerativeClustering

# For visualising high-dimensional data in 2D
from sklearn.decomposition import PCA

# Clustering evaluation metrics
from sklearn.metrics import silhouette_score, adjusted_rand_score


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Data directories
DATA_DIR = "output/processed"
PLOTS_DIR = "output/plots"

# We know we have 10 classes (A-J), so we expect 10 clusters
N_CLUSTERS = 10

# Range of k values to test for finding optimal clusters
K_RANGE = range(2, 16)


# ==============================================================================
# FUNCTION: Load Data
# ==============================================================================

def load_data():
    """
    Load the preprocessed data.

    For unsupervised learning, we REMOVE the labels, but keep them
    for evaluation (comparing clusters with true classes).

    Returns:
        tuple: (X_all, y_labels, label_mapping)
    """
    print("\nLoading data...")

    # Load training and test sets
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

    # Combine train and test (for clustering we use all data)
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    # Load label mapping
    label_mapping = {}
    mapping_file = os.path.join(DATA_DIR, 'label_mapping.txt')
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    label_mapping[int(parts[0])] = parts[1]

    # Convert numeric labels to letters
    y_labels = np.array([label_mapping.get(int(y), str(y)) for y in y_all])

    print(f"  Total samples: {len(X_all)}")
    print(f"  Features: {X_all.shape[1]}")
    print(f"  True classes: {sorted(set(y_labels))}")

    return X_all, y_labels, label_mapping


# ==============================================================================
# FUNCTION: Elbow Method
# ==============================================================================

def elbow_method(X, k_range=K_RANGE):
    """
    Use the Elbow Method to find optimal number of clusters.

    The elbow method plots the within-cluster sum of squares (inertia)
    for different values of k. The "elbow" point where the curve bends
    suggests a good number of clusters.

    Args:
        X: Feature data
        k_range: Range of k values to try

    Reference: https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
    """
    print("\n" + "-" * 50)
    print("ELBOW METHOD")
    print("-" * 50)

    inertias = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        print(f"  k={k:2d}: Inertia = {kmeans.inertia_:.2f}")

    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14)

    # Mark k=10 (our expected number of classes)
    plt.axvline(x=10, color='red', linestyle='--', label='Expected k=10')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'clustering_elbow.png'), dpi=150)
    plt.close()
    print(f"\n  Saved: clustering_elbow.png")


# ==============================================================================
# FUNCTION: Silhouette Analysis
# ==============================================================================

def silhouette_analysis(X, k_range=K_RANGE):
    """
    Use Silhouette Score to find optimal number of clusters.

    Silhouette score measures how similar points are to their own cluster
    vs other clusters. Score ranges from -1 to 1:
        - Close to 1: Points are well-matched to their cluster
        - Close to 0: Points are on boundary between clusters
        - Close to -1: Points may be in wrong cluster

    Args:
        X: Feature data
        k_range: Range of k values to try

    Returns:
        int: Best k value based on silhouette score

    Reference: https://www.geeksforgeeks.org/silhouette-algorithm-to-determine-the-optimal-value-of-k/
    """
    print("\n" + "-" * 50)
    print("SILHOUETTE ANALYSIS")
    print("-" * 50)

    scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
        print(f"  k={k:2d}: Silhouette Score = {score:.4f}")

    # Find best k
    best_idx = np.argmax(scores)
    best_k = list(k_range)[best_idx]

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), scores, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Score for Different k Values', fontsize=14)

    # Mark best k and expected k=10
    plt.axvline(x=best_k, color='green', linestyle='--',
                label=f'Best k={best_k} (score={scores[best_idx]:.4f})')
    plt.axvline(x=10, color='red', linestyle='--', label='Expected k=10')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'clustering_silhouette.png'), dpi=150)
    plt.close()

    print(f"\n  Best k by Silhouette: {best_k}")
    print(f"  Saved: clustering_silhouette.png")

    return best_k


# ==============================================================================
# FUNCTION: K-Means Clustering
# ==============================================================================

def perform_kmeans(X, y_true, n_clusters=10):
    """
    Perform K-Means clustering and compare with true labels.

    K-Means algorithm:
        1. Randomly place k cluster centres
        2. Assign each point to nearest centre
        3. Move centres to mean of assigned points
        4. Repeat until convergence

    Args:
        X: Feature data
        y_true: True labels (for comparison only)
        n_clusters: Number of clusters

    Returns:
        tuple: (cluster_labels, metrics)

    Reference: https://www.ibm.com/think/topics/k-means-clustering
    """
    print("\n" + "-" * 50)
    print("K-MEANS CLUSTERING")
    print("-" * 50)

    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # Calculate metrics
    silhouette = silhouette_score(X, cluster_labels)
    ari = adjusted_rand_score(y_true, cluster_labels)

    print(f"\n  Number of clusters: {n_clusters}")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Adjusted Rand Index: {ari:.4f}")
    print(f"    (ARI=1 means clusters perfectly match true labels)")

    return cluster_labels, {'silhouette': silhouette, 'ari': ari}


# ==============================================================================
# FUNCTION: Hierarchical Clustering
# ==============================================================================

def perform_hierarchical(X, y_true, n_clusters=10):
    """
    Perform Hierarchical (Agglomerative) clustering.

    Hierarchical clustering:
        1. Start with each point as its own cluster
        2. Repeatedly merge the two closest clusters
        3. Stop when we have the desired number of clusters

    Linkage methods:
        - 'ward': Minimises variance within clusters (usually best)
        - 'complete': Uses maximum distance between clusters
        - 'average': Uses average distance between clusters

    Args:
        X: Feature data
        y_true: True labels (for comparison only)
        n_clusters: Number of clusters

    Returns:
        tuple: (cluster_labels, metrics)

    Reference: https://www.ibm.com/think/topics/hierarchical-clustering
               https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
               https://jbhender.github.io/Stats506/F18/GP/Group10.html
    """
    print("\n" + "-" * 50)
    print("HIERARCHICAL CLUSTERING")
    print("-" * 50)

    # Fit Agglomerative Clustering with Ward linkage
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = hierarchical.fit_predict(X)

    # Calculate metrics
    silhouette = silhouette_score(X, cluster_labels)
    ari = adjusted_rand_score(y_true, cluster_labels)

    print(f"\n  Number of clusters: {n_clusters}")
    print(f"  Linkage method: ward")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Adjusted Rand Index: {ari:.4f}")

    return cluster_labels, {'silhouette': silhouette, 'ari': ari}


# ==============================================================================
# FUNCTION: Visualise Clusters with PCA
# ==============================================================================

def visualise_clusters_pca(X, cluster_labels, true_labels, title, filename):
    """
    Visualise clusters in 2D using PCA dimensionality reduction.

    Since our data has 63 dimensions, we can't plot it directly.
    PCA reduces it to 2D while preserving as much variance as possible.

    Args:
        X: Feature data
        cluster_labels: Assigned cluster labels
        true_labels: True class labels
        title: Plot title
        filename: Where to save

    Reference: https://www.geeksforgeeks.org/machine-learning/reduce-data-dimentionality-using-pca-python/
    """
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Clusters
    scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels,
                               cmap='tab10', alpha=0.6, s=20)
    axes[0].set_xlabel('PC1', fontsize=12)
    axes[0].set_ylabel('PC2', fontsize=12)
    axes[0].set_title(f'{title} - Cluster Assignments', fontsize=14)
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')

    # Plot 2: True Labels
    unique_labels = sorted(set(true_labels))
    label_to_num = {label: i for i, label in enumerate(unique_labels)}
    y_numeric = [label_to_num[label] for label in true_labels]

    scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=y_numeric,
                               cmap='tab10', alpha=0.6, s=20)
    axes[1].set_xlabel('PC1', fontsize=12)
    axes[1].set_ylabel('PC2', fontsize=12)
    axes[1].set_title('True Class Labels', fontsize=14)

    # Create legend for true labels
    handles = [plt.scatter([], [], c=[plt.cm.tab10(i/10)], label=label)
               for i, label in enumerate(unique_labels)]
    axes[1].legend(handles=handles, title='Class', loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ==============================================================================
# FUNCTION: Compare Methods
# ==============================================================================

def compare_methods(kmeans_metrics, hier_metrics):
    """
    Compare K-Means and Hierarchical clustering results.

    Args:
        kmeans_metrics: Dict with K-Means metrics
        hier_metrics: Dict with Hierarchical metrics
    """
    print("\n" + "=" * 50)
    print("CLUSTERING COMPARISON")
    print("=" * 50)

    comparison = pd.DataFrame({
        'Method': ['K-Means', 'Hierarchical (Ward)'],
        'Silhouette Score': [kmeans_metrics['silhouette'], hier_metrics['silhouette']],
        'Adjusted Rand Index': [kmeans_metrics['ari'], hier_metrics['ari']]
    })

    print("\n" + comparison.to_string(index=False))

    # Save comparison
    comparison.to_csv(os.path.join(PLOTS_DIR, 'clustering_comparison.csv'), index=False)

    # Create comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Silhouette comparison
    axes[0].bar(comparison['Method'], comparison['Silhouette Score'],
                color=['steelblue', 'coral'], alpha=0.8)
    axes[0].set_ylabel('Silhouette Score', fontsize=12)
    axes[0].set_title('Silhouette Score Comparison', fontsize=14)
    axes[0].set_ylim([0, 1])

    # ARI comparison
    axes[1].bar(comparison['Method'], comparison['Adjusted Rand Index'],
                color=['steelblue', 'coral'], alpha=0.8)
    axes[1].set_ylabel('Adjusted Rand Index', fontsize=12)
    axes[1].set_title('Adjusted Rand Index Comparison', fontsize=14)
    axes[1].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'clustering_comparison_chart.png'), dpi=150)
    plt.close()
    print(f"\n  Saved: clustering_comparison_chart.png")

    # Determine winner
    best_method = 'K-Means' if kmeans_metrics['ari'] > hier_metrics['ari'] else 'Hierarchical'
    print(f"\n  BEST METHOD (by ARI): {best_method}")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Main function - runs our complete unsupervised learning pipeline.
    """
    print("=" * 60)
    print("UNSUPERVISED LEARNING - ASL Hand Pose Clustering")
    print("Part 2d: Clustering Data")
    print("=" * 60)
    print(f"\nStart time: {datetime.now().strftime('%H:%M:%S')}")

    # Create output directory
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load data
    try:
        X, y_true, label_mapping = load_data()
    except FileNotFoundError:
        print(f"\nERROR: Data files not found in {DATA_DIR}")
        print("Please run 02_data_preprocessing.py first.")
        return

    # =========================================================================
    # Step 1: Find Optimal Number of Clusters
    # =========================================================================
    elbow_method(X)
    best_k = silhouette_analysis(X)

    # =========================================================================
    # Step 2: K-Means Clustering
    # =========================================================================
    kmeans_labels, kmeans_metrics = perform_kmeans(X, y_true, n_clusters=N_CLUSTERS)

    # Visualise K-Means results
    visualise_clusters_pca(X, kmeans_labels, y_true, 'K-Means',
                           'kmeans_pca_visualisation.png')

    # =========================================================================
    # Step 3: Hierarchical Clustering
    # =========================================================================
    hier_labels, hier_metrics = perform_hierarchical(X, y_true, n_clusters=N_CLUSTERS)

    # Visualise Hierarchical results
    visualise_clusters_pca(X, hier_labels, y_true, 'Hierarchical',
                           'hierarchical_pca_visualisation.png')

    # =========================================================================
    # Step 4: Compare Methods
    # =========================================================================
    compare_methods(kmeans_metrics, hier_metrics)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("UNSUPERVISED LEARNING COMPLETE")
    print("=" * 60)

    print("\n  Key Findings:")
    print(f"    - Expected clusters: 10 (ASL letters A-J)")
    print(f"    - Best k by Silhouette: {best_k}")
    print(f"    - K-Means ARI: {kmeans_metrics['ari']:.4f}")
    print(f"    - Hierarchical ARI: {hier_metrics['ari']:.4f}")

    print("\n  Note: ARI (Adjusted Rand Index) measures how well")
    print("  clusters match true class labels (1.0 = perfect match)")

    print(f"\n  All plots saved to: {PLOTS_DIR}")
    print(f"\nEnd time: {datetime.now().strftime('%H:%M:%S')}")


# ==============================================================================
# RUN THE SCRIPT
# ==============================================================================

if __name__ == "__main__":
    main()