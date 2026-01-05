"""
================================================================================
06_visualisation.py - Visualisation Utilities
================================================================================

Module:     CMP-6058A/7058A Artificial Intelligence
University: University of East Anglia (UEA)
Author:     [Group A108]

COURSEWORK PART 2e: Academic Poster Visualisations

This module provides visualisation utilities for the ASL hand pose
recognition project, including:
    - Results summary charts

These visualisations are designed for use in the academic poster.

LEARNING RESOURCES:
    - MediaPipe Hands: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    - Matplotlib: https://matplotlib.org/stable/contents.html

================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for saving plots

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PLOTS_DIR = "output/plots"

# ==============================================================================
# FUNCTION: Create Results Summary
# ==============================================================================

def create_results_summary(supervised_results, unsupervised_results, output_path):
    """
    Create a summary visualisation of all results.
    Useful for the academic poster.

    Args:
        supervised_results: DataFrame with classifier comparison
        unsupervised_results: DataFrame with clustering comparison
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Supervised Learning Results
    if supervised_results is not None:
        models = supervised_results['model_name']
        test_acc = supervised_results['test_accuracy']

        bars = axes[0].bar(models, test_acc, color=['#3498db', '#2ecc71', '#e74c3c'])
        axes[0].set_ylabel('Test Accuracy', fontsize=12)
        axes[0].set_title('Supervised Learning - Classifier Comparison', fontsize=14)
        axes[0].set_ylim([0, 1.1])

        # Add value labels
        for bar, acc in zip(bars, test_acc):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f'{acc:.3f}', ha='center', fontsize=11, fontweight='bold')

    # Unsupervised Learning Results
    if unsupervised_results is not None:
        methods = unsupervised_results['Method']
        ari = unsupervised_results['ARI']

        bars = axes[1].bar(methods, ari, color=['#9b59b6', '#1abc9c'])
        axes[1].set_ylabel('Adjusted Rand Index', fontsize=12)
        axes[1].set_title('Unsupervised Learning - Clustering Comparison', fontsize=14)
        axes[1].set_ylim([0, max(ari) * 1.3])

        # Add value labels
        for bar, score in zip(bars, ari):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{score:.3f}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Generate visualisation diagrams for the academic poster.
    """
    print("=" * 60)
    print("GENERATING VISUALISATION DIAGRAMS")
    print("Part 2e: Academic Poster Visualisations")
    print("=" * 60)

    # Create output directory
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("VISUALISATION COMPLETE")
    print("=" * 60)
    print(f"\n  Plots saved to: {PLOTS_DIR}")


# ==============================================================================
# RUN THE SCRIPT
# ==============================================================================

if __name__ == "__main__":
    main()