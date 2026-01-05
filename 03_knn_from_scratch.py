"""
================================================================================
03_knn_from_scratch.py - k-Nearest Neighbours from Scratch
================================================================================

Module:     CMP-6058A/7058A Artificial Intelligence
University: University of East Anglia (UEA)
Author:     [Group A108]

COURSEWORK PART 2c: kNN must be implemented from scratch using ONLY Python
                    standard built-in libraries

This script is the k-Nearest Neighbours (kNN) algorithm WITHOUT using
NumPy, scikit-learn, or any external libraries - only Python standard libraries
as specified in the brief.

HOW kNN WORKS:
    1. Calculate the distance from test point to ALL training points
    2. Find the k closest (nearest) training points
    3. Look at the labels of these k neighbours
    4. Predict the most common label (majority voting)

ALLOWED LIBRARIES (Python standard library only):
    - math: For mathematical functions like sqrt()
    - csv: For reading CSV files
    - collections: For Counter (counting votes)
    - random: For shuffling data
    - os: For file operations

LEARNING RESOURCES:
    - kNN Algorithm: https://www.geeksforgeeks.org/k-nearest-neighbours/
    - Euclidean Distance: https://www.geeksforgeeks.org/euclidean-distance/
    - 5-fold CV: https://www.w3schools.com/python/python_ml_cross_validation.asp
    - k values: https://www.geeksforgeeks.org/machine-learning/how-to-find-the-optimal-value-of-k-in-knn/

================================================================================
"""

# ==============================================================================
# IMPORTS - ONLY Python Standard Library (as required by coursework)
# ==============================================================================

import math                      # For sqrt (square root)
from collections import Counter  # For counting class votes
import csv                       # For reading CSV files
import random                    # For shuffling data
import os                        # For file operations


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Where is the cleaned data file?
DATA_FILE = "output/processed/asl_features_cleaned.csv"

# Train/test split ratio (80% train, 20% test)
TEST_RATIO = 0.2

# k values to test during cross-validation
K_VALUES = [1, 3, 5, 7, 9, 11]

# Number of folds for cross-validation (as required by coursework)
N_FOLDS = 5

# Random seed for reproducibility
RANDOM_SEED = 42


# ==============================================================================
# FUNCTION: Calculate Euclidean Distance
# ==============================================================================

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    This is the "straight line" distance - like measuring with a ruler.

    Formula: sqrt( (x1-x2)² + (y1-y2)² + ... )

    Example:
        point1 = [0, 0], point2 = [3, 4]
        distance = sqrt(3² + 4²) = sqrt(25) = 5

    Args:
        point1: List of coordinates for first point
        point2: List of coordinates for second point

    Returns:
        float: The Euclidean distance

    Reference: https://www.geeksforgeeks.org/euclidean-distance/
    """
    # Both points must have the same number of dimensions
    if len(point1) != len(point2):
        raise ValueError("Points must have same number of dimensions!")

    # Sum up the squared differences for each dimension
    squared_sum = 0.0
    for i in range(len(point1)):
        diff = point1[i] - point2[i]
        squared_sum += diff * diff

    # Return square root of the sum
    return math.sqrt(squared_sum)


# ==============================================================================
# FUNCTION: Find k Nearest Neighbours
# ==============================================================================

def find_k_nearest_neighbours(training_data, training_labels, test_point, k):
    """
    Find the k nearest neighbours to a test point.
    This is the core of our kNN algorithm.

    Steps:
        1. Calculate distance from test point to each training point
        2. Sort by distance (closest first)
        3. Return the k closest neighbours

    Args:
        training_data: List of training feature vectors
        training_labels: List of labels for training data
        test_point: The point we want to classify
        k: Number of neighbours to find

    Returns:
        list: The k nearest neighbours as (distance, label) tuples

    Reference: https://www.ibm.com/think/topics/knn
    """
    # Calculate distance to each training point
    distances = []
    for i in range(len(training_data)):
        dist = euclidean_distance(training_data[i], test_point)
        distances.append((dist, training_labels[i]))

    # Sort by distance (smallest first)
    distances.sort(key=lambda x: x[0])

    # Return the k nearest
    return distances[:k]


# ==============================================================================
# FUNCTION: Predict Single Point
# ==============================================================================

def predict_single(training_data, training_labels, test_point, k):
    """
    Predict the class for a single test point using kNN.
    Uses MAJORITY VOTING - the class with most neighbours wins.

    Why use odd values of k?
        To avoid ties! With k=3, you can't have a 1.5 vs 1.5 tie.

    Args:
        training_data: List of training feature vectors
        training_labels: List of labels for training data
        test_point: The point we want to classify
        k: Number of neighbours to consider

    Returns:
        The predicted class label
    """
    # Find k nearest neighbours
    neighbours = find_k_nearest_neighbours(training_data, training_labels, test_point, k)

    # Extract just the labels from neighbours
    neighbour_labels = [label for (distance, label) in neighbours]

    # Count votes for each class using Counter
    vote_counts = Counter(neighbour_labels)

    # Return the class with the most votes
    predicted_label = vote_counts.most_common(1)[0][0]

    return predicted_label


# ==============================================================================
# FUNCTION: Predict All Test Points
# ==============================================================================

def predict_all(training_data, training_labels, test_data, k, verbose=True):
    """
    Predict classes for multiple test points.
    Simply applies predict_single to each test point.

    Args:
        training_data: List of training feature vectors
        training_labels: List of labels for training data
        test_data: List of test feature vectors
        k: Number of neighbours
        verbose: Whether to print progress

    Returns:
        list: Predicted labels for all test points
    """
    predictions = []

    for i, test_point in enumerate(test_data):
        pred = predict_single(training_data, training_labels, test_point, k)
        predictions.append(pred)

        # Progress indicator every 50 samples
        if verbose and (i + 1) % 50 == 0:
            print(f"    Predicted {i + 1}/{len(test_data)} samples...")

    return predictions


# ==============================================================================
# FUNCTION: Calculate Accuracy
# ==============================================================================

def calculate_accuracy(true_labels, predicted_labels):
    """
    Calculate the accuracy of our predictions.

    Accuracy = (Number of correct predictions) / (Total predictions)

    Note: With imbalanced classes, accuracy alone can be misleading.
    That's why we also use SMOTE and confusion matrices.

    Args:
        true_labels: The actual correct labels
        predicted_labels: The labels our model predicted

    Returns:
        float: Accuracy as a decimal (0.0 to 1.0)
    """
    correct = 0
    for true, pred in zip(true_labels, predicted_labels):
        if true == pred:
            correct += 1

    return correct / len(true_labels)


# ==============================================================================
# FUNCTION: Create Confusion Matrix
# ==============================================================================

def create_confusion_matrix(true_labels, predicted_labels):
    """
    Create a confusion matrix to show classification results.

    A confusion matrix shows:
        - Rows = actual classes
        - Columns = predicted classes
        - Each cell = count of that combination

    Args:
        true_labels: The actual correct labels
        predicted_labels: The labels our model predicted

    Returns:
        tuple: (matrix as dict of dicts, list of unique labels)
    """
    # Get all unique labels
    all_labels = sorted(set(true_labels) | set(predicted_labels))

    # Initialise matrix with zeros
    matrix = {}
    for true_label in all_labels:
        matrix[true_label] = {}
        for pred_label in all_labels:
            matrix[true_label][pred_label] = 0

    # Fill in counts
    for true, pred in zip(true_labels, predicted_labels):
        matrix[true][pred] += 1

    return matrix, all_labels


# ==============================================================================
# FUNCTION: Print Confusion Matrix
# ==============================================================================

def print_confusion_matrix(matrix, labels):
    """
    Print the confusion matrix in a readable format.
    """
    print("\n  Confusion Matrix:")
    print("  " + "Predicted".center(len(labels) * 6))
    print("  " + "     " + "  ".join(f"{l:>4}" for l in labels))
    print("  " + "-" * (6 + len(labels) * 6))

    for true_label in labels:
        row = f"  {true_label} |"
        for pred_label in labels:
            count = matrix[true_label][pred_label]
            row += f" {count:4d} "
        print(row)


# ==============================================================================
# FUNCTION: 5-Fold Cross-Validation
# ==============================================================================

def k_fold_cross_validation(features, labels, k_neighbours, n_folds=5):
    """
    Perform k-fold cross-validation (required by coursework).

    Why cross-validation?
        We want to know how well our model will perform on NEW data.
        We split our training data into folds, train on some, test on others.

    With 5 folds, we get this pattern:
        Fold 1: Train on [2,3,4,5], Test on [1]
        Fold 2: Train on [1,3,4,5], Test on [2]
        Fold 3: Train on [1,2,4,5], Test on [3]
        Fold 4: Train on [1,2,3,5], Test on [4]
        Fold 5: Train on [1,2,3,4], Test on [5]

    Args:
        features: All feature data
        labels: All labels
        k_neighbours: The k value for kNN
        n_folds: Number of folds (default 5 as required)

    Returns:
        dict: Contains mean accuracy, std, and fold accuracies

    Reference: https://scikit-learn.org/stable/modules/cross_validation.html
    """
    # Create indices and shuffle them
    indices = list(range(len(features)))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)

    # Split indices into n_folds roughly equal parts
    fold_size = len(indices) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        # Last fold gets any remaining samples
        end = len(indices) if i == n_folds - 1 else start + fold_size
        folds.append(indices[start:end])

    # Perform cross-validation
    fold_accuracies = []

    for fold_idx in range(n_folds):
        # Test indices are the current fold
        test_indices = folds[fold_idx]

        # Train indices are all other folds
        train_indices = []
        for i in range(n_folds):
            if i != fold_idx:
                train_indices.extend(folds[i])

        # Get train and test data
        train_features = [features[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_features = [features[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]

        # Make predictions (verbose=False for CV to reduce output)
        predictions = predict_all(train_features, train_labels, test_features,
                                  k_neighbours, verbose=False)

        # Calculate accuracy
        accuracy = calculate_accuracy(test_labels, predictions)
        fold_accuracies.append(accuracy)

        print(f"    Fold {fold_idx + 1}: Accuracy = {accuracy:.4f}")

    # Calculate mean and standard deviation
    mean_acc = sum(fold_accuracies) / len(fold_accuracies)
    variance = sum((x - mean_acc) ** 2 for x in fold_accuracies) / len(fold_accuracies)
    std_acc = math.sqrt(variance)

    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'fold_accuracies': fold_accuracies
    }


# ==============================================================================
# FUNCTION: Load Data from CSV
# ==============================================================================

def load_data_from_csv(filepath):
    """
    Load feature data from CSV file using only standard library.

    Args:
        filepath: Path to the CSV file

    Returns:
        tuple: (features as list of lists, labels as list)
    """
    features = []
    labels = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header row

        # Features start after: instance_id, filename, folder
        feature_start = 3
        # Label is the last column
        label_idx = -1

        for row in reader:
            # Extract features (convert to float)
            feature_values = [float(x) for x in row[feature_start:label_idx]]
            features.append(feature_values)

            # Extract label
            labels.append(row[label_idx])

    return features, labels


# ==============================================================================
# FUNCTION: Standardise Features
# ==============================================================================

def standardise_features(features):
    """
    Standardise features to have mean=0 and std=1.

    This is important for kNN because it uses distances.
    Without standardisation, features with larger values would dominate.

    Formula: z = (x - mean) / std

    Args:
        features: List of feature vectors

    Returns:
        tuple: (standardised features, means, stds)
    """
    n_features = len(features[0])
    n_samples = len(features)

    # Calculate mean for each feature column
    means = []
    for j in range(n_features):
        col_sum = sum(features[i][j] for i in range(n_samples))
        means.append(col_sum / n_samples)

    # Calculate standard deviation for each feature
    stds = []
    for j in range(n_features):
        variance_sum = sum((features[i][j] - means[j]) ** 2 for i in range(n_samples))
        variance = variance_sum / n_samples
        stds.append(math.sqrt(variance) if variance > 0 else 1.0)

    # Standardise each value
    standardised = []
    for i in range(n_samples):
        row = []
        for j in range(n_features):
            z = (features[i][j] - means[j]) / stds[j]
            row.append(z)
        standardised.append(row)

    return standardised, means, stds


# ==============================================================================
# FUNCTION: Apply Standardisation (using pre-computed values)
# ==============================================================================

def apply_standardisation(features, means, stds):
    """
    Apply standardisation using pre-computed means and stds.
    Used to standardise test data using training data statistics.

    Args:
        features: Features to standardise
        means: Pre-computed means (from training data)
        stds: Pre-computed standard deviations (from training data)

    Returns:
        Standardised features
    """
    standardised = []
    for row in features:
        new_row = []
        for j in range(len(row)):
            z = (row[j] - means[j]) / stds[j]
            new_row.append(z)
        standardised.append(new_row)

    return standardised


# ==============================================================================
# FUNCTION: Split Data into Train/Test
# ==============================================================================

def split_data(features, labels, test_ratio=0.2):
    """
    Split data into training and test sets.

    Args:
        features: All features
        labels: All labels
        test_ratio: Fraction for test set (default 0.2 = 20%)

    Returns:
        tuple: (train_features, test_features, train_labels, test_labels)
    """
    # Create indices and shuffle
    indices = list(range(len(features)))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)

    # Calculate split point
    split_idx = int(len(indices) * (1 - test_ratio))

    # Split indices
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    # Create train and test sets
    train_features = [features[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_features = [features[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_features, test_features, train_labels, test_labels


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Main function - runs the kNN from scratch implementation.

    This implements the coursework requirements:
        1. kNN using only Python standard libraries
        2. 5-fold cross-validation for hyperparameter tuning
        3. Testing different k values
        4. Evaluation on test set
    """
    print("=" * 60)
    print("kNN FROM SCRATCH - ASL Hand Pose Classification")
    print("Part 2c: kNN using only Python standard libraries")
    print("=" * 60)
    print("\nThis implementation uses ONLY:")
    print("  math, csv, collections, random, os")

    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        print(f"\nERROR: Data file not found: {DATA_FILE}")
        print("Please run 02_data_preprocessing.py first.")
        return

    # Step 1: Load data
    print(f"\nLoading data from: {DATA_FILE}")
    features, labels = load_data_from_csv(DATA_FILE)
    print(f"  Loaded {len(features)} samples with {len(features[0])} features")
    print(f"  Classes: {sorted(set(labels))}")

    # Step 2: Split into train/test
    print("\nSplitting data (80% train, 20% test)...")
    train_features, test_features, train_labels, test_labels = split_data(
        features, labels, test_ratio=TEST_RATIO
    )
    print(f"  Training: {len(train_features)} samples")
    print(f"  Test: {len(test_features)} samples")

    # Step 3: Standardise features
    print("\nStandardising features...")
    train_std, means, stds = standardise_features(train_features)
    test_std = apply_standardisation(test_features, means, stds)
    print("  Features normalised (mean=0, std=1)")

    # ==================================================================
    # HYPERPARAMETER TUNING using 5-fold Cross-Validation
    # ==================================================================
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING (5-fold Cross-Validation)")
    print("=" * 60)
    print(f"\nTesting k values: {K_VALUES}")

    results = []
    for k in K_VALUES:
        print(f"\n  Testing k = {k}:")
        cv_result = k_fold_cross_validation(train_std, train_labels, k, n_folds=N_FOLDS)
        results.append({
            'k': k,
            'mean_accuracy': cv_result['mean_accuracy'],
            'std_accuracy': cv_result['std_accuracy']
        })
        print(f"    Mean CV Accuracy: {cv_result['mean_accuracy']:.4f} "
              f"(+/- {cv_result['std_accuracy']:.4f})")

    # Find best k
    best_result = max(results, key=lambda x: x['mean_accuracy'])
    best_k = best_result['k']

    print("\n" + "-" * 50)
    print("BEST HYPERPARAMETERS")
    print("-" * 50)
    print(f"  Best k = {best_k}")
    print(f"  CV Accuracy = {best_result['mean_accuracy']:.4f}")

    # ==================================================================
    # FINAL EVALUATION ON TEST SET
    # ==================================================================
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    print(f"\n  Making predictions with k = {best_k}...")
    predictions = predict_all(train_std, train_labels, test_std, best_k)

    # Calculate accuracy
    test_accuracy = calculate_accuracy(test_labels, predictions)
    print(f"\n  TEST ACCURACY: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Show confusion matrix
    matrix, labels_list = create_confusion_matrix(test_labels, predictions)
    print_confusion_matrix(matrix, labels_list)

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Best k value: {best_k}")
    print(f"  Cross-validation accuracy: {best_result['mean_accuracy']:.4f}")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    print("\n  This kNN was implemented from scratch using only:")
    print("  math, csv, collections, random, os")


# ==============================================================================
# RUN THE SCRIPT
# ==============================================================================

if __name__ == "__main__":
    main()