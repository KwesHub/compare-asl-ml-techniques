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
    https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn
    4. Random Forest (our third classifier choice)

WHY THESE CLASSIFIERS?

    Decision Tree (REQUIRED):
        - Simple, interpretable classifier
        - Creates a tree of if-then rules based on features
        - Provides feature importance rankings
        - Can overfit without proper pruning (controlled via max_depth)
        Reference: https://scikit-learn.org/stable/modules/tree.html

    kNN from Scratch (REQUIRED):
        - Demonstrates understanding of algorithm mechanics
        - Imported from 03_knn_from_scratch.py
        - Validates our implementation against sklearn
        Reference: See 03_knn_from_scratch.py documentation

    kNN (sklearn):
        - Professional implementation for comparison
        - Faster execution due to optimised code
        - Validates our from-scratch implementation
        Reference: https://scikit-learn.org/stable/modules/neighbors.html

    Random Forest (OUR CHOICE):
        - Ensemble of multiple decision trees
        - More robust than single decision tree
        - Reduces overfitting through averaging
        - Provides feature importance
        Reference: https://scikit-learn.org/stable/modules/ensemble.html#forest
        https://www.kaggle.com/code/prashant111/random-forest-classifier-tutorial

FOR EACH CLASSIFIER:
    1. Define hyperparameter search space
    2. Perform 5-fold cross-validation grid search
    3. Find optimal hyperparameters
    4. Train final model on full training set
    5. Evaluate on test set
    6. Generate visualisations (confusion matrices, feature importance)

    We use GridSearchCV with 5-fold cross-validation to find optimal
    hyperparameters for each classifier. This ensures:
    - Fair comparison (all classifiers tuned similarly)
    - estimates (5-fold averaging)
    - Prevention of overfitting to training data
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    For each classifier, we test the default values plus alternatives,
    as required by the coursework brief.

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
# This validates that our implementation works correctly against sklearn, we wanted to make sure
# ours was correct and felt this was the best way to have a kNN face off...lol

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
# FUNCTION: Load Preprocessed Data
# ==============================================================================

def load_data():
    """
    Load our preprocessed data from numpy files created by 02_data_preprocessing.py.

    The preprocessing pipeline (02_data_preprocessing.py) saves:
        - X_train.npy: Training features (SMOTE-balanced, standardised)
        - X_test.npy: Test features (standardised using training statistics)
        - y_train.npy: Training labels (encoded as integers 0-9)
        - y_test.npy: Test labels (encoded as integers 0-9)
        - label_mapping.txt: Maps integers back to letters (0→A, 1→B, etc.)

    This separation ensures:
        1. Preprocessing is done once, used many times (efficiency)
        2. Consistent data across all experiments (reproducibility)
        3. Proper separation of training and test data (no data leakage)

    Returns:
        tuple: (X_train, X_test, y_train, y_test, label_mapping)
            - X_train: numpy array of shape (n_train, 63)
            - X_test: numpy array of shape (n_test, 63)
            - y_train: numpy array of shape (n_train,) with integer labels
            - y_test: numpy array of shape (n_test,) with integer labels
            - label_mapping: dict mapping integers to letter labels
    """
    #Pipeline update
    print("\nLoading preprocessed data...")

    # Load our numpy arrays containing features and labels
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

    # Load label mapping (number -> letter)
    # This allows us to convert predictions back to readable letters
    label_mapping = {}
    mapping_file = os.path.join(DATA_DIR, 'label_mapping.txt')
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    label_mapping[int(parts[0])] = parts[1]

    # Feedback showing dataset characteristics
    print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test: {X_test.shape[0]} samples")
    print(f"  Classes: {list(label_mapping.values())}")

    return X_train, X_test, y_train, y_test, label_mapping


# ==============================================================================
# VISUALISATION FUNCTIONS
# ==============================================================================

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    """
    Create and save a confusion matrix heatmap visualisation.

    A confusion matrix shows the relationship between true and predicted classes:
        - Rows represent actual/true classes
        - Columns represent predicted classes
        - Diagonal elements = correct predictions
        - Off-diagonal elements = miss-classifications

    The heatmap colour intensity shows the count magnitude:
        - Darker blue = more samples
        - Lighter = fewer samples

    This visualisation helps identify:
        - Which classes are well-classified (strong diagonal)
        - Which classes are confused with each other (off-diagonal clusters)
        - Overall classifier performance patterns

    Parameters:
        y_true (array): True class labels (ground truth)
        y_pred (array): Predicted class labels from the model
        labels (list): Class label names for axis labels (A, B, C, etc.)
        title (str): Title for the plot
        filename (str): Filename to save the plot (will be saved in PLOTS_DIR)
    """
    # Calculate confusion matrix using sklearn
    cm = confusion_matrix(y_true, y_pred)

    # Create heatmap visualisation
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()

    # Save to output directory
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_feature_importance(model, feature_names, title, filename, top_n=15):
    """
    Plot feature importance for tree-based models (Decision Tree, Random Forest).

    Feature importance shows which features (hand landmarks) contribute most
    to the classification decisions. This helps us understand:
        - Which parts of the hand are most distinctive for ASL letters
        - Whether certain landmarks are more informative than others
        - Potential for feature selection/dimensionality reduction in the future. our thought
        is that there may be key features that indicate the ASL sign not all of them.

    Tree-based models compute importance as the total reduction in impurity.

    Parameters:
        model: Trained sklearn model with feature_importances_ attribute
        feature_names (list): Names of features (e.g., 'landmark_0_x', etc.)
        title (str): Title for the plot
        filename (str): Filename to save the plot
        top_n (int): Number of top features to display (default 15)

    Note: Only works for models with feature_importances_ attribute
          (Decision Tree, Random Forest, etc.)

    Reference: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    """
    # Check if model has feature importance attribute
    if not hasattr(model, 'feature_importances_'):
        print(f"  Note: {type(model).__name__} doesn't have feature_importances_")
        return

    # Get importance's and sort in descending order
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]  # Top n features

    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), importances[indices], color='steelblue', alpha=0.8)

    # Set x-axis labels (feature names or indices)
    if feature_names:
        plt.xticks(range(top_n), [feature_names[i] for i in indices],
                   rotation=45, ha='right')
    else:
        plt.xticks(range(top_n), [f'Feature {i}' for i in indices], rotation=45)

    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ==============================================================================
# CLASSIFIER 1: DECISION TREE
# ==============================================================================

def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree classifier with hyperparameter tuning.

    Decision Trees work by creating a tree of if-then rules based on features:
        - Each internal node tests a feature condition
        - Each branch represents the outcome of a test
        - Each leaf node holds a class prediction

    Hyperparameters we tune:

        max_depth (int or None):
            - Maximum depth of the tree
            - None = nodes are expanded until all leaves are pure
            - Smaller values prevent overfitting but may underfit
            - We test: None (default), 5, 10, 15, 20

        min_samples_split (int):
            - Minimum samples required to split an internal node
            - Larger values prevent overfitting by avoiding small splits
            - Default is 2 (can split down to 2 samples)
            - We test: 2 (default), 5, 10, 20

    Why these hyperparameters?
        - They control tree complexity and generalisation
        - They directly impact overfitting vs underfitting trade-off
        - They are commonly tuned for decision trees

    Parameters:
        X_train (array): Training feature vectors
        y_train (array): Training labels

    Returns:
        tuple: (best_model, best_params, cv_score)
            - best_model: Trained DecisionTreeClassifier with optimal params
            - best_params: Dictionary of optimal hyperparameter values
            - cv_score: Best cross-validation accuracy achieved

    Reference: https://scikit-learn.org/stable/modules/tree.html
    """
    # Pipeline update
    print("\n" + "-" * 50)
    print("DECISION TREE CLASSIFIER")
    print("-" * 50)

    # Define hyperparameter search space
    # Including None/2 as defaults (required by coursework)
    param_grid = {
        'max_depth': [None, 5, 10, 15, 20],  # None = unlimited depth
        'min_samples_split': [2, 5, 10, 20]  # 2 = default minimum
    }

    print(f"\n  Hyperparameters to tune:")
    print(f"    max_depth: {param_grid['max_depth']}")
    print(f"    min_samples_split: {param_grid['min_samples_split']}")
    print(f"  Total combinations: {len(param_grid['max_depth']) * len(param_grid['min_samples_split'])}")

    # Create base Decision Tree model with fixed random state for reproducibility
    dt = DecisionTreeClassifier(random_state=42)

    # Set up stratified 5-fold cross-validation
    # Stratified ensures each fold has representative class distribution
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    # Perform grid search with cross-validation
    # n_jobs=-1 uses all available CPU cores for parallel computation
    print(f"\n  Running {CV_FOLDS}-fold cross-validation grid search...")
    grid_search = GridSearchCV(dt, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Report results
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
    Train kNN using our from-scratch implementation (03_knn_from_scratch.py).

    This fulfils the coursework requirement for kNN implementation using
    ONLY Python standard libraries (math, collections, csv, random, os).

    The actual algorithm code is in 03_knn_from_scratch.py. This function
    serves as a wrapper that:
        1. Converts numpy arrays to Python lists
        2. Calls our from-scratch cross-validation function
        3. Returns results in a format consistent with sklearn classifiers

    Hyperparameters tuned:
        k (n_neighbours): Number of neighbours for voting
            - We test: 1, 3, 5, 7, 9, 11
            - Odd values avoid ties in majority voting

    Note: Our implementation is slower than sklearn because:
        - Pure Python vs optimised C/Cython code
        - No vectorised operations (no numpy)
        - No efficient data structures (no KD-trees)

    But it demonstrates understanding of the algorithm!

    Parameters:
        X_train (array): Training feature vectors
        y_train (array): Training labels

    Returns:
        tuple: (best_k, best_params, cv_score)
            - best_k: Optimal k value found
            - best_params: Dictionary {'k': best_k}
            - cv_score: Best cross-validation accuracy achieved

    Reference: See 03_knn_from_scratch.py for full implementation details
    """
    # Pipeline update
    print("\n" + "-" * 50)
    print("kNN FROM SCRATCH (Python standard library only)")
    print("-" * 50)

    # Check if our from-scratch module was successfully imported
    if not KNN_SCRATCH_AVAILABLE:
        print("\n  ERROR: kNN from scratch module not available!")
        print("  Please ensure 03_knn_from_scratch.py is in the same directory.")
        return None, {}, 0.0

    print("\n  Using functions from 03_knn_from_scratch.py")
    print("  Libraries used: math, collections.Counter, csv, random, os")
    print("  NO NumPy or sklearn!")

    # Convert numpy arrays to Python lists for our from-scratch implementation
    # Our implementation uses only standard Python data structures
    X_train_list = X_train.tolist()
    y_train_list = y_train.tolist()

    # k values to test (same as sklearn for fair comparison)
    k_values = [1, 3, 5, 7, 9, 11]

    print(f"\n  Hyperparameters to tune:")
    print(f"    k (n_neighbours): {k_values}")
    print(f"\n  Running {CV_FOLDS}-fold cross-validation...")
    print("  (This may take longer than sklearn due to pure Python implementation)")

    # Test each k value using our from-scratch cross-validation
    results = []
    for k in k_values:
        # Call our from-scratch CV function
        cv_result = k_fold_cross_validation(X_train_list, y_train_list, k,
                                            n_folds=CV_FOLDS)
        results.append({
            'k': k,
            'mean_accuracy': cv_result['mean_accuracy'],
            'std_accuracy': cv_result['std_accuracy']
        })
        print(f"    k={k}: CV Accuracy = {cv_result['mean_accuracy']:.4f} "
              f"(± {cv_result['std_accuracy']:.4f})")

    # Find best k based on highest mean accuracy
    best_result = max(results, key=lambda x: x['mean_accuracy'])
    best_k = best_result['k']

    print(f"\n  Best Parameters:")
    print(f"    k: {best_k}")
    print(f"  Best CV Accuracy: {best_result['mean_accuracy']:.4f}")

    return best_k, {'k': best_k}, best_result['mean_accuracy']


def evaluate_knn_from_scratch(X_train, X_test, y_train, y_test, best_k, label_mapping):
    """
    Evaluate our from-scratch kNN implementation on the test set.

    This function:
        1. Makes predictions on the test set using optimal k
        2. Calculates test accuracy
        3. Estimates training accuracy (sampled for speed)
        4. Creates confusion matrix visualisation

    Note on training accuracy sampling:
        Predicting on all training samples with kNN from scratch is slow
        (O(n²) for n samples). We sample 500 training points for a
        reasonable estimate of training accuracy.

    Parameters:
        X_train, X_test: Feature arrays (numpy)
        y_train, y_test: Label arrays (numpy)
        best_k: Optimal k value from cross-validation
        label_mapping: Dict mapping numbers to letters

    Returns:
        dict: Results containing:
            - 'name': Classifier name
            - 'train_accuracy': Estimated training accuracy
            - 'test_accuracy': Test set accuracy
            - 'predictions': Test set predictions
    """
    print(f"\n  Evaluating kNN (from scratch) with k={best_k}...")

    if not KNN_SCRATCH_AVAILABLE:
        return {'name': 'kNN (from scratch)', 'train_accuracy': 0,
                'test_accuracy': 0, 'predictions': []}

    # Convert numpy arrays to Python lists
    X_train_list = X_train.tolist()
    X_test_list = X_test.tolist()
    y_train_list = y_train.tolist()
    y_test_list = y_test.tolist()

    # Predict on test set
    print(f"    Making predictions on {len(X_test_list)} test samples...")
    test_pred = predict_all(X_train_list, y_train_list, X_test_list, best_k,
                            verbose=True)

    # Calculate test accuracy
    correct = sum(1 for true, pred in zip(y_test_list, test_pred) if true == pred)
    test_acc = correct / len(y_test_list)

    # Estimate training accuracy using a sample (for speed)
    # Full training prediction would be O(n²) which is slow
    print(f"    Calculating training accuracy (sampling 500 points for speed)...")
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

    # Report accuracies
    print(f"    Training Accuracy (sampled): {train_acc:.4f}")
    print(f"    Test Accuracy:               {test_acc:.4f}")

    # Create confusion matrix visualisation
    labels = [label_mapping[i] for i in sorted(label_mapping.keys())]
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

    This professional implementation lets us validate our from-scratch
    version by comparing results. sklearn's kNN is much faster because:
        - Implemented in optimised C/Cython code
        - Uses vectorised numpy operations
        - Employs efficient data structures (KD-trees, Ball trees)

    Hyperparameters we tune:

        n_neighbors (int):
            - Number of neighbours for voting (the "k" in kNN)
            - We test: 1, 3, 5, 7, 9, 11 (same as from-scratch)

        weights (str):
            - 'uniform': All neighbours vote equally
            - 'distance': Closer neighbours have more influence
            - Distance weighting can improve accuracy when neighbours
              vary significantly in their distance from the test point

    Parameters:
        X_train (array): Training feature vectors
        y_train (array): Training labels

    Returns:
        tuple: (best_model, best_params, cv_score)
            - best_model: Trained KNeighborsClassifier with optimal params
            - best_params: Dictionary of optimal hyperparameter values
            - cv_score: Best cross-validation accuracy achieved

    Reference: https://scikit-learn.org/stable/modules/neighbors.html
    """
    # Pipeline update
    print("\n" + "-" * 50)
    print("kNN CLASSIFIER (sklearn) - For Comparison")
    print("-" * 50)

    # Define hyperparameter search space
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11],  # Same k values as from-scratch
        'weights': ['uniform', 'distance']  # Voting weight strategy
    }

    print(f"\n  Hyperparameters to tune:")
    print(f"    n_neighbors (k): {param_grid['n_neighbors']}")
    print(f"    weights: {param_grid['weights']}")
    print(f"  Total combinations: {len(param_grid['n_neighbors']) * len(param_grid['weights'])}")

    # Create base kNN model
    knn = KNeighborsClassifier()

    # Set up stratified cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    # Perform grid search
    print(f"\n  Running {CV_FOLDS}-fold cross-validation grid search...")
    grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Report results
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

    Random Forest is an ENSEMBLE method - it combines multiple decision trees
    to make more robust predictions. The key ideas are:

        1. Bootstrap Aggregating (Bagging):
           - Each tree is trained on a random sample of the data
           - Samples are drawn with replacement (bootstrap)

        2. Random Feature Selection:
           - At each split, only a random subset of features is considered
           - Reduces correlation between trees

        3. Majority Voting:
           - Each tree makes a prediction
           - Final prediction is the majority vote across all trees

    Why Random Forest?
        - More robust than single Decision Tree (reduces overfitting)
        - Handles high-dimensional data well (63 features)
        - Provides feature importance rankings
        - Generally achieves good accuracy without extensive tuning

    Hyperparameters we tune:

        n_estimators (int):
            - Number of trees in the forest
            - More trees = better accuracy but slower
            - We test: 50, 100, 150

        max_depth (int or None):
            - Maximum depth of each tree
            - Same meaning as Decision Tree
            - We test: None, 10, 20

    Parameters:
        X_train (array): Training feature vectors
        y_train (array): Training labels

    Returns:
        tuple: (best_model, best_params, cv_score)

    Reference: https://scikit-learn.org/stable/modules/ensemble.html#forest
    """
    # Pipeline update
    print("\n" + "-" * 50)
    print("RANDOM FOREST CLASSIFIER (Our Choice)")
    print("-" * 50)

    # Define hyperparameter search space
    param_grid = {
        'n_estimators': [50, 100, 150],  # Number of trees
        'max_depth': [None, 10, 20]  # Tree depth limit
    }

    print(f"\n  Hyperparameters to tune:")
    print(f"    n_estimators: {param_grid['n_estimators']}")
    print(f"    max_depth: {param_grid['max_depth']}")
    print(f"  Total combinations: {len(param_grid['n_estimators']) * len(param_grid['max_depth'])}")

    # Create base Random Forest model
    rf = RandomForestClassifier(random_state=42)

    # Set up cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    # Perform grid search
    print(f"\n  Running {CV_FOLDS}-fold cross-validation grid search...")
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Report results
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

    This function provides a standardised evaluation for all sklearn classifiers:
        1. Predicts on training set (to check for overfitting)
        2. Predicts on test set (true performance estimate)
        3. Calculates accuracy for both
        4. Creates confusion matrix visualisation

    Training vs Test Accuracy:
        - If train >> test: Model is overfitting
        - If train ≈ test: Model generalises well
        - If both low: Model is underfitting

    Parameters:
        model: Trained sklearn classifier
        X_train, X_test: Feature arrays
        y_train, y_test: Label arrays
        name (str): Classifier name for display/filenames
        label_mapping (dict): Maps integers to letter labels

    Returns:
        dict: Contains:
            - 'name': Classifier name
            - 'train_accuracy': Training set accuracy
            - 'test_accuracy': Test set accuracy
            - 'predictions': Test set predictions
    """
    print(f"\n  Evaluating {name}...")

    # Predict on training set
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    # Predict on test set
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    # Report accuracies
    print(f"    Training Accuracy: {train_acc:.4f}")
    print(f"    Test Accuracy:     {test_acc:.4f}")

    # Check for overfitting
    if train_acc - test_acc > 0.1:
        print(f"    WARNING: Large train-test gap may indicate overfitting")

    # Create confusion matrix plot
    labels = [label_mapping[i] for i in sorted(label_mapping.keys())]
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
    Create a comprehensive comparison of all classifiers.

    This function:
        1. Creates a comparison table (printed and saved to CSV)
        2. Creates a bar chart comparing train/test accuracies
        3. Identifies the best overall classifier
        4. Compares kNN implementations (from-scratch vs sklearn)

    The comparison helps answer:
        - Which classifier performs best on this dataset?
        - Are classifiers overfitting (large train-test gap)?
        - How does our from-scratch kNN compare to sklearn?

    Parameters:
        results (list): List of result dictionaries from evaluate_model()
    """
    # Pipeline update
    print("\n" + "=" * 50)
    print("CLASSIFIER COMPARISON")
    print("=" * 50)

    # Create comparison DataFrame for easy viewing
    comparison = pd.DataFrame([
        {
            'Classifier': r['name'],
            'Train Accuracy': r['train_accuracy'],
            'Test Accuracy': r['test_accuracy']
        }
        for r in results
    ])

    # Print comparison table
    print("\n" + comparison.to_string(index=False))

    # Save to CSV for reports
    comparison.to_csv(os.path.join(PLOTS_DIR, 'classifier_comparison.csv'), index=False)
    print(f"\n  Saved: classifier_comparison.csv")

    # Create comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(comparison))
    width = 0.35

    # Create grouped bars
    bars1 = ax.bar(x - width / 2, comparison['Train Accuracy'], width,
                   label='Train Accuracy', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width / 2, comparison['Test Accuracy'], width,
                   label='Test Accuracy', color='coral', alpha=0.8)

    # Formatting
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Classifier Comparison - ASL Hand Pose Recognition', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison['Classifier'], rotation=15, ha='right')
    ax.legend()
    ax.set_ylim([0.85, 1.05])  # Start at 0.85 to better show differences

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'classifier_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: classifier_comparison.png")

    # Identify best classifier by test accuracy
    best_idx = comparison['Test Accuracy'].idxmax()
    print(f"\n  BEST CLASSIFIER: {comparison.loc[best_idx, 'Classifier']}")
    print(f"  Test Accuracy: {comparison.loc[best_idx, 'Test Accuracy']:.4f}")

    # Compare kNN implementations
    knn_scratch = comparison[comparison['Classifier'] == 'kNN (from scratch)']['Test Accuracy'].values
    knn_sklearn = comparison[comparison['Classifier'] == 'kNN (sklearn)']['Test Accuracy'].values

    if len(knn_scratch) > 0 and len(knn_sklearn) > 0:
        print(f"\n  kNN IMPLEMENTATION COMPARISON:")
        print(f"    From scratch (pure Python): {knn_scratch[0]:.4f}")
        print(f"    sklearn (optimised):        {knn_sklearn[0]:.4f}")
        diff = abs(knn_scratch[0] - knn_sklearn[0])
        print(f"    Difference:                 {diff:.4f}")

        if diff < 0.02:
            print("    Implementations are consistent (difference < 2%)")
        else:
            print("    Note: Difference may be due to different random splits")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Main function - runs the complete supervised learning pipeline.

    Coursework requirements for Part 2c:
        1. Decision Tree classifier (required)
        2. kNN from scratch (required)
        3. kNN sklearn + Random Forest (additional classifiers we used sklearn version to make sure ours was
        working well and Random forest is our additional.)
        4. 5-fold cross-validation for hyperparameter tuning
        5. Comparison of all classifiers
        6. Visualisations (confusion matrices, feature importance)
    """
    # Pipeline header
    print("=" * 60)
    print("SUPERVISED LEARNING - ASL Hand Pose Classification")
    print("Part 2c: Optimising, evaluating and comparing classifiers")
    print("=" * 60)
    print(f"\nStart time: {datetime.now().strftime('%H:%M:%S')}")

    # Create output directory for plots
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ==========================================================================
    # Load preprocessed data
    # ==========================================================================
    try:
        X_train, X_test, y_train, y_test, label_mapping = load_data()
    except FileNotFoundError:
        print(f"\nERROR: Data files not found in {DATA_DIR}")
        print("Please run 02_data_preprocessing.py first.")
        return

    # Load feature names for feature importance plots
    feature_names = []
    feat_file = os.path.join(DATA_DIR, 'feature_names.txt')
    if os.path.exists(feat_file):
        with open(feat_file, 'r') as f:
            feature_names = [line.strip() for line in f]

    # Store results for final comparison
    all_results = []

    # ==========================================================================
    # CLASSIFIER 1: Decision Tree (REQUIRED)
    # ==========================================================================
    dt_model, dt_params, dt_cv = train_decision_tree(X_train, y_train)
    dt_result = evaluate_model(dt_model, X_train, X_test, y_train, y_test,
                               'Decision Tree', label_mapping)
    dt_result['cv_accuracy'] = dt_cv
    dt_result['best_params'] = dt_params
    all_results.append(dt_result)

    # Feature importance for Decision Tree
    plot_feature_importance(dt_model, feature_names,
                            'Decision Tree - Feature Importance (Top 15)',
                            'decision_tree_feature_importance.png')

    # ==========================================================================
    # CLASSIFIER 2: kNN FROM SCRATCH (REQUIRED)
    # ==========================================================================
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
        #error handling while we worked on separate modules
        print("\n  Skipping kNN from scratch (module not available)")

    # ==========================================================================
    # CLASSIFIER 3: kNN (sklearn) - For comparison
    # ==========================================================================
    knn_model, knn_params, knn_cv = train_knn_sklearn(X_train, y_train)
    knn_result = evaluate_model(knn_model, X_train, X_test, y_train, y_test,
                                'kNN (sklearn)', label_mapping)
    knn_result['cv_accuracy'] = knn_cv
    knn_result['best_params'] = knn_params
    all_results.append(knn_result)

    # ==========================================================================
    # CLASSIFIER 4: Random Forest (Our choice)
    # ==========================================================================
    rf_model, rf_params, rf_cv = train_random_forest(X_train, y_train)
    rf_result = evaluate_model(rf_model, X_train, X_test, y_train, y_test,
                               'Random Forest', label_mapping)
    rf_result['cv_accuracy'] = rf_cv
    rf_result['best_params'] = rf_params
    all_results.append(rf_result)

    # Feature importance for Random Forest
    plot_feature_importance(rf_model, feature_names,
                            'Random Forest - Feature Importance (Top 15)',
                            'random_forest_feature_importance.png')

    # ==========================================================================
    # Compare all classifiers
    # ==========================================================================
    compare_classifiers(all_results)

    # ==========================================================================
    # Summary
    # ==========================================================================
    #Pipeline update complete
    print("\n" + "=" * 60)
    print("SUPERVISED LEARNING COMPLETE")
    print("=" * 60)

    print("\n  BEST HYPERPARAMETERS FOUND:")
    for result in all_results:
        print(f"\n    {result['name']}:")
        if 'best_params' in result:
            for param, value in result['best_params'].items():
                print(f"      {param}: {value}")
        if 'cv_accuracy' in result:
            print(f"      CV Accuracy: {result['cv_accuracy']:.4f}")
        print(f"      Test Accuracy: {result['test_accuracy']:.4f}")

    print(f"\n  All visualisations saved to: {PLOTS_DIR}")
    print(f"\nEnd time: {datetime.now().strftime('%H:%M:%S')}")


# ==============================================================================
# RUN THE SCRIPT
# ==============================================================================

if __name__ == "__main__":
    main()