"""
================================================================================
06_visualisation.py - Visualisation Utilities for Academic Poster
================================================================================

Module:     CMP-6058A/7058A Artificial Intelligence
University: University of East Anglia (UEA)
Author:     [Group A108]

COURSEWORK PART 2e: Academic Poster Visualisations

This module provides utility functions for creating our high-quality visualisations
suitable for the academic poster and presentation. It consolidates key results
from supervised and unsupervised learning into summary graphics.

LEARNING RESOURCES:
    - Matplotlib Documentation: https://matplotlib.org/stable/contents.html
    - Seaborn for statistical plots: https://seaborn.pydata.org/
    - UEA Library Poster Resources: https://my.uea.ac.uk/divisions/library-and-learning-enhancement

================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

# Use non-interactive backend BEFORE importing pyplot
# This prevents display issues when saving plots on headless systems
# https://matplotlib.org/3.2.2/tutorials/introductory/usage.html
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # Core plotting library
import os                        # For file path operations


# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Settings for visualisation output and styling.

# Output directory for all our generated plots
PLOTS_DIR = "output/plots"

# Default figure DPI for high-quality output
# 300 DPI is standard for print quality and our A1 poster versions
DEFAULT_DPI = 300

# Colour palette suitable for colour-blind viewers
# Using colours that are distinguishable by most people
COLOUR_PALETTE = {
    'primary': '#3498db',      # Blue
    'secondary': '#2ecc71',    # Green
    'tertiary': '#e74c3c',     # Red
    'quaternary': '#9b59b6',   # Purple
    'quinary': '#1abc9c'       # Teal
}


# ==============================================================================
# FUNCTION: Create Results Summary Visualisation
# ==============================================================================

def create_results_summary(supervised_results, unsupervised_results, output_path):
    """
    Create a summary visualisation combining supervised and unsupervised results.

    This creates a single figure with two panels:
        1. Top panel: Supervised learning classifier comparison (test accuracy)
        2. Bottom panel: Unsupervised learning clustering comparison (ARI)

    This is ideal for the academic poster as it provides a quick overview
    of all experimental results in one compact figure.

    Parameters:
        supervised_results (DataFrame or None): DataFrame with columns:
            - 'model_name': Name of the classifier
            - 'test_accuracy': Test set accuracy

        unsupervised_results (DataFrame or None): DataFrame with columns:
            - 'Method': Clustering method name
            - 'ARI': Adjusted Rand Index score

        output_path (str): Full path where to save the figure

    Returns:
        None (saves our figure to file)

    """
    # Create figure with two subplots stacked vertically
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # ==========================================================================
    # Top panel: Supervised Learning Results
    # ==========================================================================
    if supervised_results is not None and len(supervised_results) > 0:
        models = supervised_results['model_name']
        test_acc = supervised_results['test_accuracy']

        # Define colours for each classifier
        colours = [COLOUR_PALETTE['primary'],
                   COLOUR_PALETTE['secondary'],
                   COLOUR_PALETTE['tertiary']]

        # Extend colours if more than 3 classifiers
        while len(colours) < len(models):
            colours.extend([COLOUR_PALETTE['quaternary'], COLOUR_PALETTE['quinary']])

        bars = axes[0].bar(models, test_acc, color=colours[:len(models)], alpha=0.85)
        axes[0].set_ylabel('Test Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_title('Supervised Learning - Classifier Comparison',
                          fontsize=16, fontweight='bold', pad=15)
        axes[0].set_ylim([0, 1.1])  # Accuracy ranges 0-1

        # Add horizontal line at perfect accuracy for reference
        axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
                        label='Perfect Accuracy')

        # Add value labels on top of each bar
        for bar, acc in zip(bars, test_acc):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f'{acc:.3f}', ha='center', fontsize=12, fontweight='bold')

        # Style improvements
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
    else:
        axes[0].text(0.5, 0.5, 'No supervised learning results available',
                     ha='center', va='center', fontsize=14)
        axes[0].set_title('Supervised Learning Results', fontsize=16)

    # ==========================================================================
    # Bottom panel: Unsupervised Learning Results
    # ==========================================================================
    if unsupervised_results is not None and len(unsupervised_results) > 0:
        methods = unsupervised_results['Method']
        ari = unsupervised_results['ARI']

        # Define colours for clustering methods
        colours = [COLOUR_PALETTE['quaternary'], COLOUR_PALETTE['quinary']]

        bars = axes[1].bar(methods, ari, color=colours[:len(methods)], alpha=0.85)
        axes[1].set_ylabel('Adjusted Rand Index (ARI)', fontsize=14, fontweight='bold')
        axes[1].set_title('Unsupervised Learning - Clustering Comparison',
                          fontsize=16, fontweight='bold', pad=15)

        # Set y-axis limit with some headroom for labels
        axes[1].set_ylim([0, max(ari) * 1.3])

        # Add value labels on top of each bar
        for bar, score in zip(bars, ari):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{score:.3f}', ha='center', fontsize=12, fontweight='bold')

        # Style improvements
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
    else:
        axes[1].text(0.5, 0.5, 'No unsupervised learning results available',
                     ha='center', va='center', fontsize=14)
        axes[1].set_title('Unsupervised Learning Results', fontsize=16)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"  Saved summary visualisation: {output_path}")


# ==============================================================================
# FUNCTION: Create Poster-Ready Accuracy Comparison
# ==============================================================================

def create_poster_accuracy_chart(classifiers, accuracies, output_path):
    """
    Create a large, poster-ready classifier accuracy comparison chart.

    This creates a horizontal bar chart optimised for poster display:
        - Large fonts readable from distance
        - High contrast colours
        - Clean, minimal design

    Parameters:
        classifiers (list): List of classifier names
        accuracies (list): Corresponding accuracy values (0-1 range)
        output_path (str): Where to save the figure

    Returns:
        None (saves our figure to file)
    """
    # Create figure - wider for horizontal bars
    fig, ax = plt.subplots(figsize=(14, 8))

    # Sort by accuracy (best at top)
    sorted_indices = sorted(range(len(accuracies)), key=lambda i: accuracies[i])
    sorted_classifiers = [classifiers[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]

    # Create horizontal bar chart
    colours = plt.cm.viridis([i/len(classifiers) for i in range(len(classifiers))])
    bars = ax.barh(sorted_classifiers, sorted_accuracies, color=colours, alpha=0.85)

    # Add value labels at the end of each bar
    for bar, acc in zip(bars, sorted_accuracies):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.1%}', va='center', fontsize=14, fontweight='bold')

    # Formatting
    ax.set_xlabel('Accuracy', fontsize=18, fontweight='bold')
    ax.set_title('Classifier Performance Comparison', fontsize=22, fontweight='bold', pad=20)
    ax.set_xlim([0, 1.15])

    # Add vertical line at perfect accuracy
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)

    # Make y-tick labels larger
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=12)

    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"  Saved poster accuracy chart: {output_path}")

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Main function - generates visualisation diagrams for the academic poster.

    This script can be run standalone to demonstrate the visualisation
    functions, or its functions can be imported by other scripts.
    """
    # Pipeline update
    print("=" * 60)
    print("VISUALISATION UTILITIES")
    print("Part 2e: Academic Poster Visualisations")
    print("=" * 60)

    # Create output directory
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Note: Our Actual visualisations are created by the other scripts
    # This module provides utility functions that can be imported

    # Pipeline update
    print("\n" + "=" * 60)
    print("VISUALISATION MODULE LOADED")
    print("=" * 60)
    print(f"\n  Output directory: {PLOTS_DIR}")
    print(f"  Default DPI: {DEFAULT_DPI}")


# ==============================================================================
# RUN THE SCRIPT
# ==============================================================================

if __name__ == "__main__":
    main()