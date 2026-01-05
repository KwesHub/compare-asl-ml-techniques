"""
================================================================================
04_supervised_learning.py - Supervised Learning Classifiers
================================================================================

Module:     CMP-6058A/7058A Artificial Intelligence
University: University of East Anglia (UEA)
Author:     [Group A108]

COURSEWORK PART 2c: Supervised Learning - Optimising, evaluating and comparing
                    classifiers

This script trains and compares FOUR supervised learning classifiers:
    1. Decision Tree (REQUIRED by coursework)
    2. kNN from scratch (REQUIRED - imported from 03_knn_from_scratch.py)
    3. kNN using scikit-learn (for comparison with our from-scratch version)
    4. Random Forest (our third classifier choice)

FOR EACH CLASSIFIER:
    1. Tune hyperparameters using 5-fold cross-validation
    2. Find the best hyperparameter settings
    3. Train final model on full training set
    4. Evaluate on test set
    5. Create visualisations (confusion matrices, feature importance)

LEARNING RESOURCES:
    - Decision Trees: https://scikit-learn.org/stable/modules/tree.html
    - kNN: https://scikit-learn.org/stable/modules/neighbors.html
    - Random Forest: https://scikit-learn.org/stable/modules/ensemble.html#forest
    - Cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html

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
import seaborn as sns
import os
from datetime import datetime

# Scikit-learn classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model selection and evaluation
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

# ==============================================================================
# IMPORT kNN FROM SCRATCH (from 03_knn_from_scratch.py)
# ==============================================================================
# We import our from-scratch implementation to include it in the comparison.
# This validates that our implementation works correctly against sklearn.

try:
    from importlib import import_module
    knn_scratch_module = import_module('03_knn_from_scratch')

    # Import the functions we need
    predict_all = knn_scratch_module.predict_all
    k_fold_cross_validation = knn_scratch_module.k_fold_cross_validation

    KNN_SCRATCH_AVAILABLE = True
    print("Successfully imported kNN from scratch (03_knn_from_scratch.py)")
except ImportError as e:
    KNN_SCRATCH_AVAILABLE = False
    print(f"WARNING: Could not import kNN from scratch: {e}")
    print("         kNN from scratch comparison will be skipped.")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Data directories
DATA_DIR = "output/processed"
PLOTS_DIR = "output/plots"

# Cross-validation settings (as required by coursework)
CV_FOLDS = 5


# ==============================================================================
# FUNCTION: Load Data
# ==============================================================================

def load_data():
    """
    Load the preprocessed data from numpy files.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, label_mapping)
    """
    print("\nLoading preprocessed data...")

    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

    # Load label mapping (number → letter)
    label_mapping = {}
    mapping_file = os.path.join(DATA_DIR, 'label_mapping.txt')
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    label_mapping[int(parts[0])] = parts[1]

    print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test: {X_test.shape[0]} samples")
    print(f"  Classes: {list(label_mapping.values())}")

    return X_train, X_test, y_train, y_test, label_mapping


# ==============================================================================
# VISUALISATION FUNCTIONS
# ==============================================================================

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    """
    Create and save a confusion matrix plot.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels
        title: Plot title
        filename: Where to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_feature_importance(model, feature_names, title, filename, top_n=15):
    """
    Plot feature importance for tree-based models.
    Shows which features are most useful for classification.

    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        title: Plot title
        filename: Where to save
        top_n: Number of top features to show
    """
    if not hasattr(model, 'feature_importances_'):
        return

    # Get importances and sort
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), importances[indices], color='steelblue', alpha=0.8)

    # Use feature indices if names not available
    if feature_names:
        plt.xticks(range(top_n), [feature_names[i] for i in indices],
                   rotation=45, ha='right')
    else:
        plt.xticks(range(top_n), [f'F{i}' for i in indices], rotation=45)

    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ==============================================================================
# CLASSIFIER 1: DECISION TREE
# ==============================================================================

def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree classifier with hyperparameter tuning.

    Hyperparameters we tune:
        - max_depth: How deep the tree can grow (None = unlimited)
        - min_samples_split: Minimum samples needed to split a node

    Returns:
        tuple: (best_model, best_params, cv_score)

    Reference: https://scikit-learn.org/stable/modules/tree.html
    """
    print("\n" + "-" * 50)
    print("DECISION TREE CLASSIFIER")
    print("-" * 50)

    # Hyperparameters to search (including defaults as required by coursework)
    param_grid = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 20]
    }

    print(f"\n  Hyperparameters to tune:")
    print(f"    max_depth: {param_grid['max_depth']}")
    print(f"    min_samples_split: {param_grid['min_samples_split']}")

    # Create base model
    dt = DecisionTreeClassifier(random_state=42)

    # Set up 5-fold cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    # Grid search
    print(f"\n  Running {CV_FOLDS}-fold cross-validation...")
    grid_search = GridSearchCV(dt, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Results
    print(f"\n  Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"    {param}: {value}")
    print(f"  Best CV Accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


# ==============================================================================
# CLASSIFIER 2: kNN FROM SCRATCH
# ==============================================================================

def train_knn_from_scratch(X_train, y_train):
    """
    Train kNN using our from-scratch implementation.
    This fulfils the coursework requirement for kNN using only Python standard libraries.

    Note: The actual kNN code is in 03_knn_from_scratch.py

    Returns:
        tuple: (best_k, best_params, cv_score)
    """
    print("\n" + "-" * 50)
    print("kNN FROM SCRATCH (Python standard library only)")
    print("-" * 50)

    if not KNN_SCRATCH_AVAILABLE:
        print("\n  ERROR: kNN from scratch module not available!")
        print("  Please ensure 03_knn_from_scratch.py is in the same directory.")
        return None, {}, 0.0

    print("\n  Using functions from 03_knn_from_scratch.py")
    print("  (Uses only: math, collections.Counter)")

    # Convert numpy arrays to Python lists for our from-scratch implementation
    X_train_list = X_train.tolist()
    y_train_list = y_train.tolist()

    # k values to test
    k_values = [1, 3, 5, 7, 9, 11]

    print(f"\n  Hyperparameters to tune:")
    print(f"    k (n_neighbours): {k_values}")
    print(f"\n  Running {CV_FOLDS}-fold cross-validation...")

    # Test each k value
    results = []
    for k in k_values:
        cv_result = k_fold_cross_validation(X_train_list, y_train_list, k, n_folds=CV_FOLDS)
        results.append({
            'k': k,
            'mean_accuracy': cv_result['mean_accuracy'],
            'std_accuracy': cv_result['std_accuracy']
        })
        print(f"    k={k}: CV Accuracy = {cv_result['mean_accuracy']:.4f} "
              f"(+/- {cv_result['std_accuracy']:.4f})")

    # Find best k
    best_result = max(results, key=lambda x: x['mean_accuracy'])
    best_k = best_result['k']

    print(f"\n  Best Parameters:")
    print(f"    k: {best_k}")
    print(f"  Best CV Accuracy: {best_result['mean_accuracy']:.4f}")

    return best_k, {'k': best_k}, best_result['mean_accuracy']


def evaluate_knn_from_scratch(X_train, X_test, y_train, y_test, best_k, label_mapping):
    """
    Evaluate kNN from scratch on the test set.

    Args:
        X_train, X_test: Feature arrays
        y_train, y_test: Label arrays
        best_k: Optimal k value from cross-validation
        label_mapping: Dict mapping numbers to letters

    Returns:
        dict: Results including accuracies and predictions
    """
    print(f"\n  Evaluating kNN (from scratch) with k={best_k}...")

    if not KNN_SCRATCH_AVAILABLE:
        return {'name': 'kNN (from scratch)', 'train_accuracy': 0,
                'test_accuracy': 0, 'predictions': []}

    # Convert to lists
    X_train_list = X_train.tolist()
    X_test_list = X_test.tolist()
    y_train_list = y_train.tolist()
    y_test_list = y_test.tolist()

    # Predict on test set
    print(f"    Making predictions on {len(X_test_list)} test samples...")
    test_pred = predict_all(X_train_list, y_train_list, X_test_list, best_k, verbose=True)

    # Calculate test accuracy
    correct = sum(1 for true, pred in zip(y_test_list, test_pred) if true == pred)
    test_acc = correct / len(y_test_list)

    # For training accuracy (sample to save time)
    print(f"    Calculating training accuracy (sampling 500 points)...")
    import random
    sample_size = min(500, len(X_train_list))
    random.seed(42)
    sample_indices = random.sample(range(len(X_train_list)), sample_size)

    train_sample_X = [X_train_list[i] for i in sample_indices]
    train_sample_y = [y_train_list[i] for i in sample_indices]

    train_pred_sample = predict_all(X_train_list, y_train_list, train_sample_X,
                                    best_k, verbose=False)
    train_correct = sum(1 for true, pred in zip(train_sample_y, train_pred_sample)
                       if true == pred)
    train_acc = train_correct / len(train_sample_y)

    print(f"    Training Accuracy (sampled): {train_acc:.4f}")
    print(f"    Test Accuracy:     {test_acc:.4f}")

    # Get class labels for confusion matrix
    labels = [label_mapping[i] for i in sorted(label_mapping.keys())]

    # Create confusion matrix plot
    plot_confusion_matrix(
        y_test_list, test_pred, labels,
        'kNN (from scratch) - Confusion Matrix',
        'knn_from_scratch_confusion_matrix.png'
    )

    return {
        'name': 'kNN (from scratch)',
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'predictions': test_pred
    }


# ==============================================================================
# CLASSIFIER 3: kNN (sklearn) - FOR COMPARISON
# ==============================================================================

def train_knn_sklearn(X_train, y_train):
    """
    Train a kNN classifier using scikit-learn for comparison.
    This lets us validate our from-scratch implementation.

    Hyperparameters we tune:
        - n_neighbours: The k value (number of neighbours)
        - weights: 'uniform' or 'distance'

    Returns:
        tuple: (best_model, best_params, cv_score)

    Reference: https://scikit-learn.org/stable/modules/neighbors.html
    """
    print("\n" + "-" * 50)
    print("kNN CLASSIFIER (sklearn) - For Comparison")
    print("-" * 50)

    # Hyperparameters to search
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance']
    }

    print(f"\n  Hyperparameters to tune:")
    print(f"    n_neighbours (k): {param_grid['n_neighbors']}")
    print(f"    weights: {param_grid['weights']}")

    # Create base model
    knn = KNeighborsClassifier()

    # Set up cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    # Grid search
    print(f"\n  Running {CV_FOLDS}-fold cross-validation...")
    grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Results
    print(f"\n  Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"    {param}: {value}")
    print(f"  Best CV Accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


# ==============================================================================
# CLASSIFIER 4: RANDOM FOREST
# ==============================================================================

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier with hyperparameter tuning.

    Random Forest is an ensemble method - it trains many decision trees
    and combines their predictions (like asking many experts and voting).

    Hyperparameters we tune:
        - n_estimators: Number of trees in the forest
        - max_depth: Maximum depth of each tree

    Returns:
        tuple: (best_model, best_params, cv_score)

    Reference: https://scikit-learn.org/stable/modules/ensemble.html#forest
    """
    print("\n" + "-" * 50)
    print("RANDOM FOREST CLASSIFIER")
    print("-" * 50)

    # Hyperparameters to search
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20]
    }

    print(f"\n  Hyperparameters to tune:")
    print(f"    n_estimators: {param_grid['n_estimators']}")
    print(f"    max_depth: {param_grid['max_depth']}")

    # Create base model
    rf = RandomForestClassifier(random_state=42)

    # Set up cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    # Grid search
    print(f"\n  Running {CV_FOLDS}-fold cross-validation...")
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Results
    print(f"\n  Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"    {param}: {value}")
    print(f"  Best CV Accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


# ==============================================================================
# MODEL EVALUATION
# ==============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, name, label_mapping):
    """
    Evaluate a trained sklearn model on training and test sets.

    Args:
        model: Trained sklearn model
        X_train, X_test: Feature arrays
        y_train, y_test: Label arrays
        name: Name of the classifier
        label_mapping: Dict mapping numbers to letters

    Returns:
        dict: Contains train accuracy, test accuracy, predictions
    """
    print(f"\n  Evaluating {name}...")

    # Predict on training set
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    # Predict on test set
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"    Training Accuracy: {train_acc:.4f}")
    print(f"    Test Accuracy:     {test_acc:.4f}")

    # Get class labels for confusion matrix
    labels = [label_mapping[i] for i in sorted(label_mapping.keys())]

    # Create confusion matrix plot
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plot_confusion_matrix(
        y_test, test_pred, labels,
        f'{name} - Confusion Matrix',
        f'{safe_name}_confusion_matrix.png'
    )

    return {
        'name': name,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'predictions': test_pred
    }


# ==============================================================================
# COMPARE ALL CLASSIFIERS
# ==============================================================================

def compare_classifiers(results):
    """
    Create a comparison chart of all classifiers.

    Args:
        results: List of result dictionaries from evaluate_model
    """
    print("\n" + "=" * 50)
    print("CLASSIFIER COMPARISON")
    print("=" * 50)

    # Create comparison DataFrame
    comparison = pd.DataFrame([
        {
            'Classifier': r['name'],
            'Train Accuracy': r['train_accuracy'],
            'Test Accuracy': r['test_accuracy']
        }
        for r in results
    ])

    print("\n" + comparison.to_string(index=False))

    # Save to CSV
    comparison.to_csv(os.path.join(PLOTS_DIR, 'classifier_comparison.csv'), index=False)

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(comparison))
    width = 0.35

    bars1 = ax.bar(x - width/2, comparison['Train Accuracy'], width,
                   label='Train Accuracy', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, comparison['Test Accuracy'], width,
                   label='Test Accuracy', color='coral', alpha=0.8)

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Classifier Comparison (Including kNN from Scratch)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison['Classifier'], rotation=15, ha='right')
    ax.legend()
    ax.set_ylim([0.85, 1.05])

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'classifier_comparison.png'), dpi=150)
    plt.close()
    print(f"\n  Saved: classifier_comparison.png")

    # Find best classifier
    best_idx = comparison['Test Accuracy'].idxmax()
    print(f"\n  BEST CLASSIFIER: {comparison.loc[best_idx, 'Classifier']}")
    print(f"  Test Accuracy: {comparison.loc[best_idx, 'Test Accuracy']:.4f}")

    # Compare kNN implementations
    knn_scratch = comparison[comparison['Classifier'] == 'kNN (from scratch)']['Test Accuracy'].values
    knn_sklearn = comparison[comparison['Classifier'] == 'kNN (sklearn)']['Test Accuracy'].values
    if len(knn_scratch) > 0 and len(knn_sklearn) > 0:
        print(f"\n  kNN IMPLEMENTATION COMPARISON:")
        print(f"    From scratch: {knn_scratch[0]:.4f}")
        print(f"    sklearn:      {knn_sklearn[0]:.4f}")
        print(f"    Difference:   {abs(knn_scratch[0] - knn_sklearn[0]):.4f}")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Main function - runs the complete supervised learning pipeline.
    """
    print("=" * 60)
    print("SUPERVISED LEARNING - ASL Hand Pose Classification")
    print("Part 2c: Optimising, evaluating and comparing classifiers")
    print("=" * 60)
    print(f"\nStart time: {datetime.now().strftime('%H:%M:%S')}")

    # Create output directory
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load data
    try:
        X_train, X_test, y_train, y_test, label_mapping = load_data()
    except FileNotFoundError:
        print(f"\nERROR: Data files not found in {DATA_DIR}")
        print("Please run 02_data_preprocessing.py first.")
        return

    # Load feature names for plotting
    feature_names = []
    feat_file = os.path.join(DATA_DIR, 'feature_names.txt')
    if os.path.exists(feat_file):
        with open(feat_file, 'r') as f:
            feature_names = [line.strip() for line in f]

    # Store results for comparison
    all_results = []

    # =========================================================================
    # CLASSIFIER 1: Decision Tree
    # =========================================================================
    dt_model, dt_params, dt_cv = train_decision_tree(X_train, y_train)
    dt_result = evaluate_model(dt_model, X_train, X_test, y_train, y_test,
                               'Decision Tree', label_mapping)
    dt_result['cv_accuracy'] = dt_cv
    dt_result['best_params'] = dt_params
    all_results.append(dt_result)

    # Feature importance for Decision Tree
    plot_feature_importance(dt_model, feature_names,
                            'Decision Tree - Feature Importance',
                            'decision_tree_feature_importance.png')

    # =========================================================================
    # CLASSIFIER 2: kNN FROM SCRATCH (Required by coursework)
    # =========================================================================
    if KNN_SCRATCH_AVAILABLE:
        best_k, knn_scratch_params, knn_scratch_cv = train_knn_from_scratch(X_train, y_train)
        if best_k is not None:
            knn_scratch_result = evaluate_knn_from_scratch(
                X_train, X_test, y_train, y_test, best_k, label_mapping
            )
            knn_scratch_result['cv_accuracy'] = knn_scratch_cv
            knn_scratch_result['best_params'] = knn_scratch_params
            all_results.append(knn_scratch_result)
    else:
        print("\n  Skipping kNN from scratch (module not available)")

    # =========================================================================
    # CLASSIFIER 3: kNN (sklearn) - For comparison
    # =========================================================================
    knn_model, knn_params, knn_cv = train_knn_sklearn(X_train, y_train)
    knn_result = evaluate_model(knn_model, X_train, X_test, y_train, y_test,
                                'kNN (sklearn)', label_mapping)
    knn_result['cv_accuracy'] = knn_cv
    knn_result['best_params'] = knn_params
    all_results.append(knn_result)

    # =========================================================================
    # CLASSIFIER 4: Random Forest
    # =========================================================================
    rf_model, rf_params, rf_cv = train_random_forest(X_train, y_train)
    rf_result = evaluate_model(rf_model, X_train, X_test, y_train, y_test,
                               'Random Forest', label_mapping)
    rf_result['cv_accuracy'] = rf_cv
    rf_result['best_params'] = rf_params
    all_results.append(rf_result)

    # Feature importance for Random Forest
    plot_feature_importance(rf_model, feature_names,
                            'Random Forest - Feature Importance',
                            'random_forest_feature_importance.png')

    # =========================================================================
    # Compare all classifiers
    # =========================================================================
    compare_classifiers(all_results)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    print("\n  Best Hyperparameters Found:")
    for result in all_results:
        print(f"\n    {result['name']}:")
        if 'best_params' in result:
            for param, value in result['best_params'].items():
                print(f"      {param}: {value}")

    print(f"\n  All plots saved to: {PLOTS_DIR}")
    print(f"\nEnd time: {datetime.now().strftime('%H:%M:%S')}")


# ==============================================================================
# RUN THE SCRIPT
# ==============================================================================

if __name__ == "__main__":
    main()