"""
================================================================================
03_knn_from_scratch.py - k-Nearest Neighbours Implementation from Scratch
================================================================================

Module:     CMP-6058A/7058A Artificial Intelligence
University: University of East Anglia (UEA)
Author:     [Group A108]

COURSEWORK PART 2c: kNN must be implemented from scratch using ONLY Python
                    standard built-in libraries

This script implements our k-Nearest Neighbours (kNN) algorithm WITHOUT using
NumPy, scikit-learn, or any external libraries - only Python standard libraries
as specified in the coursework brief.

WHAT THIS SCRIPT DOES:
    1. Loads cleaned feature data from CSV (output from preprocessing)
    2. Splits data into training (80%) and test (20%) sets
    3. Standardises features to have mean=0 and std=1
    4. Performs 5-fold cross-validation to find optimal k value
    5. Evaluates the final model on the test set
    6. Displays confusion matrix and accuracy metrics

HOW kNN WORKS:
    The kNN algorithm is a simple, non-parametric classifier
    https://www.stat.cmu.edu/~larry/=sml/nonparclass.pdf
    that works by:
    1. Calculating the distance from a test point to ALL training points
    2. Finding the k closest (nearest) training points
    3. Looking at the labels of these k neighbours
    4. Predicting the most common label (majority voting)

    The key insight we got is that similar data points (in feature space) tend to
    have similar labels. By finding the nearest neighbours, we can predict
    the class of an unknown sample based on what its neighbours are.

WHY kNN FROM SCRATCH?
    The coursework requires implementing kNN using only Python standard
    libraries to demonstrate our understanding of the mechanics.
    This implementation uses:
    - math: For sqrt() in distance calculations
    - collections.Counter: For majority voting
    - csv: For reading data files
    - random: For shuffling and splitting data
    - os: For file path operations

    Reference: Python Standard Library documentation
    https://docs.python.org/3.12/library/index.html

DISTANCE METRIC:
    We use Euclidean distance, which measures the "straight line" distance
    between two points in n-dimensional space:
    
    distance = sqrt( sum( (x_i - y_i)^2 ) ) for all dimensions i
    
    Reference: https://www.geeksforgeeks.org/euclidean-distance/

HYPERPARAMETER k:
    The value of k determines how many neighbours vote on the classification.
    - Small k (e.g., 1): More sensitive to noise, can overfit
    - Large k (e.g., 15): More stable but may miss local patterns
    - Odd k values: Recommended to avoid ties in binary classification
    
    We test k values [1, 3, 5, 7, 9, 11] using cross-validation to find
    the optimal setting for our ASL dataset.
    
    Reference: https://www.geeksforgeeks.org/machine-learning/how-to-find-the-optimal-value-of-k-in-knn/

CROSS-VALIDATION:
    5-fold cross-validation splits training data into 5 parts:
    - Train on 4 parts, validate on 1 part
    - Repeat 5 times (each part gets used as validation once)
    - Average the 5 accuracy scores
    
    This gives a more reliable estimate of model performance than a single
    train/validation split.
    
    Reference: https://www.w3schools.com/python/python_ml_cross_validation.asp

LEARNING RESOURCES:
    - kNN Algorithm: https://www.geeksforgeeks.org/k-nearest-neighbours/
    - IBM kNN Guide: https://www.ibm.com/think/topics/knn
    - Euclidean Distance: https://www.geeksforgeeks.org/euclidean-distance/
    - 5-fold CV: https://scikit-learn.org/stable/modules/cross_validation.html
    - Python Standard Library: https://docs.python.org/3.12/library/index.html

================================================================================
"""

# ==============================================================================
# IMPORTS - ONLY Python Standard Library (as required by coursework)
# ==============================================================================
# We are restricted to using only Python's built-in libraries for the kNN
# implementation. This demonstrates understanding of the algorithm without
# relying on pre-built ML libraries.

import math                      # For sqrt() - calculating Euclidean distance
from collections import Counter  # For counting class votes in majority voting
import csv                       # For reading the cleaned CSV data file
import random                    # For shuffling data and creating random splits
import os                        # For file path operations and checking existence


# ==============================================================================
# CONFIGURATION
# ==============================================================================
# These settings control the behaviour of our kNN implementation.
# We centralise them here for easy modification and consistency.

# Our cleaned data file, This comes from 02_data_preprocessing.py
# We use the cleaned and normalised CSV rather than the raw numpy arrays
# because obviously our from-scratch implementation cannot read .npy files without NumPy.
DATA_FILE = "output/processed/asl_features_cleaned.csv"

# Train/test split ratio (80% train, 20% test)
# This follows the standard practice justified in 02_data_preprocessing.py
# Reference: Gholamy et al. (2018) - https://scholarworks.utep.edu/cs_techrep/1209
TEST_RATIO = 0.2

# k values to test during cross-validation,
# We test odd values to avoid ties in majority voting.
# The Range from 1 to 11 gives us a good spread to find the optimal k as mentioned avoiding ties in binary classification.
# Reference: https://www.geeksforgeeks.org/machine-learning/how-to-find-the-optimal-value-of-k-in-knn/
K_VALUES = [1, 3, 5, 7, 9, 11]

# Number of folds for cross-validation (as required by coursework brief)
# 5-fold is a standard choice balancing bias and variance in the estimate.
# https://machinelearningmastery.com/k-fold-cross-validation/
N_FOLDS = 5

# Random seed for reproducibility
# Using the same seed ensures we get identical splits every time,
# making our results reproducible across different runs.
RANDOM_SEED = 42


# ==============================================================================
# FUNCTION: Calculate Euclidean Distance
# ==============================================================================

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points in n-dimensional space.
    
    This is the "straight line" distance, also known as L2 distance.
    https://www.geeksforgeeks.org/maths/euclidean-distance/
    It measures how far apart two points are in the feature space.
    
    Formula: distance = sqrt( sum( (x_i - y_i)^2 ) )
    
    Example:
        For 2D points: point1 = [0, 0], point2 = [3, 4]
        distance = sqrt((3-0)² + (4-0)²) = sqrt(9 + 16) = sqrt(25) = 5
        
        This is the classic 3-4-5 right triangle!
    
    Why Euclidean Distance?
        - Most intuitive measure of distance
        - Works well when features are on similar scales (which is why
          we standardise our data first)
        - Computationally more efficient
    
    Parameters:
        point1 (list): List of coordinates for the first point
        point2 (list): List of coordinates for the second point
                       Both points must have the same number of dimensions
    
    Returns:
        float: The Euclidean distance between the two points
    
    Raises:
        ValueError: If points have different number of dimensions
    """
    # Validation: Both points must have the same number of dimensions
    # For our ASL data, this should always be 63 (21 landmarks × 3 coordinates)
    if len(point1) != len(point2):
        raise ValueError(f"Points must have same dimensions! "
                        f"Got {len(point1)} and {len(point2)}")

    # Calculate sum of squared differences for each dimension
    # We iterate through each feature and compute (x_i - y_i)²
    squared_sum = 0.0
    for i in range(len(point1)):
        diff = point1[i] - point2[i]  # Difference in dimension i
        squared_sum += diff * diff     # Add squared difference to total

    # Return the square root of the sum (completing the Euclidean formula)
    return math.sqrt(squared_sum)


# ==============================================================================
# FUNCTION: Find k Nearest Neighbours
# ==============================================================================

def find_k_nearest_neighbours(training_data, training_labels, test_point, k):
    """
    Find the k nearest neighbours to a given test point from the training set.
    
    This is the core operation of the kNN algorithm. For each test point,
    we need to find which training points are closest to it.
    
    Algorithm Steps:
        1. Calculate distance from test_point to EVERY training point
        2. Store each distance along with its corresponding label
        3. Sort all distances from smallest to largest
        4. Return the k smallest distances (the k nearest neighbours)
    
    Time Complexity: O(n log n) where n is the number of training samples
        - O(n) to calculate all distances
        - O(n log n) to sort the distances
    
    This is a "brute force" approach that works well for smaller datasets
    like our ASL data. For very large datasets, more efficient methods
    like KD-trees or Ball trees are used (but not required here).
    
    Parameters:
        training_data (list): List of training feature vectors (each is a list of 63 floats)
        training_labels (list): List of labels corresponding to training data
        test_point (list): The feature vector we want to classify
        k (int): Number of neighbours to find
    
    Returns:
        list: The k nearest neighbours as (distance, label) tuples,
              sorted by distance (closest first)
    
    Reference: https://www.ibm.com/think/topics/knn
    """
    # Step 1: Calculate distance from test_point to each training point
    # We create a list of (distance, label) tuples
    distances = []
    
    for i in range(len(training_data)):
        # Calculate Euclidean distance to this training point
        dist = euclidean_distance(training_data[i], test_point)
        
        # Store the distance along with the training point's label
        # We need the label for voting later
        distances.append((dist, training_labels[i]))

    # Step 2: Sort by distance (smallest/closest first)
    # The key=lambda x: x[0] tells Python to sort by the first element
    # of each tuple (the distance), not the label
    distances.sort(key=lambda x: x[0])

    # Step 3: Return only the k nearest neighbours
    # These are the first k elements after sorting
    return distances[:k]


# ==============================================================================
# FUNCTION: Predict Single Point
# ==============================================================================

def predict_single(training_data, training_labels, test_point, k):
    """
    Predict the class label for a single test point using kNN.
    
    This function combines finding neighbours with majority voting:
    1. Find the k nearest neighbours to the test point
    2. Count how many neighbours belong to each class
    3. Return the class with the most votes (majority wins)
    
    Majority Voting:
        If k=5 and the neighbours have labels [A, A, B, A, C], then:
        - A has 3 votes
        - B has 1 vote
        - C has 1 vote
        Therefore, we predict class A (the majority vote!).
    
    Why use odd k values?
        To avoid ties! With k=4, you could have a 2-2 tie.
        With k=5, the maximum tie is 2-2-1, which still has a winner.
        Note: For multi-class problems (like our 10 ASL letters), ties
        are less common but odd k is still good practice.
    
    Parameters:
        training_data (list): List of training feature vectors
        training_labels (list): List of labels for training data
        test_point (list): The feature vector to classify
        k (int): Number of neighbours to consider for voting
    
    Returns:
        The predicted class label (string, e.g., 'A', 'B', etc.)
    
    Reference: https://www.geeksforgeeks.org/k-nearest-neighbours/
    """
    # Step 1: Find the k nearest neighbours
    neighbours = find_k_nearest_neighbours(training_data, training_labels, 
                                           test_point, k)

    # Step 2: Extract just the labels from the neighbours
    # neighbours is a list of (distance, label) tuples
    # We only need the labels for voting
    neighbour_labels = [label for (distance, label) in neighbours]

    # Step 3: Count votes for each class using Counter
    # Counter is a dictionary subclass that counts occurrences
    # Example: Counter(['A', 'A', 'B']) gives {'A': 2, 'B': 1}
    vote_counts = Counter(neighbour_labels)

    # Step 4: Get the class with the most votes
    # most_common(1) returns a list of the 1 most common element as [(label, count)]
    # We extract just the label with [0][0]
    predicted_label = vote_counts.most_common(1)[0][0]

    return predicted_label


# ==============================================================================
# FUNCTION: Predict All Test Points
# ==============================================================================

def predict_all(training_data, training_labels, test_data, k, verbose=True):
    """
    Predict class labels for multiple test points.
    
    This function simply applies predict_single() to each test point.
    It's a wrapper that simply handles attempts and progress reporting.
    
    Parameters:
        training_data (list): List of training feature vectors
        training_labels (list): List of labels for training data
        test_data (list): List of test feature vectors to classify
        k (int): Number of neighbours for kNN
        verbose (bool): If True, print progress updates every 50 samples.
                        Set to False during cross-validation to reduce output.
    
    Returns:
        list: Predicted labels for all test points, in the same order
              as the input test_data
    """
    predictions = []

    for i, test_point in enumerate(test_data):
        # Predict the class for this single test point
        pred = predict_single(training_data, training_labels, test_point, k)
        predictions.append(pred)

        # Progress indicator (every 50 samples) for user feedback
        # This helps us know the algorithm is working, especially
        # on larger datasets where prediction takes time
        if verbose and (i + 1) % 50 == 0:
            print(f"    Predicted {i + 1}/{len(test_data)} samples...")

    return predictions


# ==============================================================================
# FUNCTION: Calculate Accuracy
# ==============================================================================

def calculate_accuracy(true_labels, predicted_labels):
    """
    Calculate the accuracy of predictions.
    
    Accuracy is the simplest classification metric:
    
        Accuracy = (Number of correct predictions) / (Total predictions)
    
    For example, if we predict 90 out of 100 samples correctly:
        Accuracy = 90/100 = 0.90 = 90%
    
    Limitations of Accuracy:
        With imbalanced classes, accuracy can be misleading.
        For example, if 90% of samples are class A, a classifier that
        always predicts A would have 90% accuracy but be useless.
        
        That's why our preprocessing applies SMOTE for class balancing,
        and we also examine confusion matrices for a fuller picture,
        this is all part of our consideration of algorithmic bias.
    
    Parameters:
        true_labels (list): The actual correct labels (ground truth)
        predicted_labels (list): The labels our model predicted
    
    Returns:
        float: Accuracy as a decimal between 0.0 and 1.0
               Multiply by 100 to get percentage
    """
    # Count how many predictions match the true labels
    correct = 0
    for true, pred in zip(true_labels, predicted_labels):
        if true == pred:
            correct += 1

    # Calculate and return accuracy
    return correct / len(true_labels)


# ==============================================================================
# FUNCTION: Create Confusion Matrix
# ==============================================================================

def create_confusion_matrix(true_labels, predicted_labels):
    """
    Create a confusion matrix to visualise classification results. we wanted an inline
    matrix to really show the results during the process which helped greatly with
    testing. we are after a nice diagonal majority to show our classification was working
    well.
    
    A confusion matrix is a table that shows:
        - Rows = actual/true classes
        - Columns = predicted classes
        - Each cell = count of samples with that (true, predicted) combination
    
    Example for 3 classes (A, B, C):
                    Predicted
                    A    B    C
        True  A  [ 45   3    2 ]   <- 45 A's correctly classified
              B  [  2  48    0 ]   <- 48 B's correctly classified
              C  [  1   0   49 ]   <- 49 C's correctly classified
    
    Reading the matrix:
        - Diagonal elements = correct predictions
        - Off-diagonal elements = misclassifications
        - Row A, Column B = A's misclassified as B
    
    This helps identify which classes are being confused with each other.
    For ASL, we might find that similar hand poses (like I and J) are
    frequently confused, especially if static not dynamic images..
    
    Parameters:
        true_labels (list): The actual correct labels
        predicted_labels (list): The labels our model predicted
    
    Returns:
        tuple: (matrix, labels) where:
            - matrix: dict of dicts, matrix[true][pred] = count
            - labels: sorted list of unique class labels
    """
    # Get all unique labels from both true and predicted
    # Using set union (|) ensures we capture all labels even if some
    # don't appear in predictions
    # https://www.w3schools.com/python/ref_set_union.asp#:~:text=The%20union()%20method%20returns,the%20specified%20set(s).
    all_labels = sorted(set(true_labels) | set(predicted_labels))

    # Initialise the matrix as a nested dictionary with zeros
    # matrix[true_label][predicted_label] will hold the count
    matrix = {}
    for true_label in all_labels:
        matrix[true_label] = {}
        for pred_label in all_labels:
            matrix[true_label][pred_label] = 0

    # Fill in the counts by iterating through all predictions
    for true, pred in zip(true_labels, predicted_labels):
        matrix[true][pred] += 1

    return matrix, all_labels


# ==============================================================================
# FUNCTION: Print Confusion Matrix
# ==============================================================================

def print_confusion_matrix(matrix, labels):
    """
    Print the confusion matrix in a readable tabular format.
    
    The output format shows:
        - Column headers = predicted labels
        - Row headers = true labels
        - Cell values = counts
    
    Example output:
              Predicted
              A    B    C    D    E    F    G    H    I    J
        --------------------------------------------------------
        A |   45    1    0    2    0    0    0    1    0    1
        B |    0   48    0    0    2    0    0    0    0    0
        ... etc.
    
    Parameters:
        matrix (dict): Confusion matrix from create_confusion_matrix()
        labels (list): Sorted list of class labels
    """
    # Print header row showing "Predicted" and column labels
    print("\n  Confusion Matrix:")
    print("  " + "Predicted".center(len(labels) * 6))
    print("  " + "     " + "  ".join(f"{l:>4}" for l in labels))
    print("  " + "-" * (6 + len(labels) * 6))

    # Print each row (one per true label) we wanted to keep naming obvious too
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
    Perform k-fold cross-validation to evaluate a k value for kNN.
    
    Cross-validation is essential for hyperparameter tuning because:
    1. It uses all data for both training and validation
    2. It provides a more reliable estimate than a single split
    3. It helps detect overfitting
    
    How 5-fold CV works:
        The data is split into 5 equal parts (folds).
        We perform 5 iterations:
        
        Iteration 1: Train on folds [2,3,4,5], Validate on fold [1]
        Iteration 2: Train on folds [1,3,4,5], Validate on fold [2]
        Iteration 3: Train on folds [1,2,4,5], Validate on fold [3]
        Iteration 4: Train on folds [1,2,3,5], Validate on fold [4]
        Iteration 5: Train on folds [1,2,3,4], Validate on fold [5]
        
        We then average the 5 validation accuracies to get the final score.
    
    Why 5 folds?
        - 5 or 10 folds are standard choices
        - Balances bias (too few folds) vs variance (too many folds)
        - 5 folds means each fold has 20% of the data, providing
          a reasonable validation set size
    
    Parameters:
        features (list): All feature vectors (training data)
        labels (list): All corresponding labels
        k_neighbours (int): The k value to evaluate
        n_folds (int): Number of folds (default 5 as required by coursework)
    
    Returns:
        dict: Contains:
            - 'mean_accuracy': Average accuracy across all folds
            - 'std_accuracy': Standard deviation of fold accuracies
            - 'fold_accuracies': List of individual fold accuracies
    
    Reference: https://scikit-learn.org/stable/modules/cross_validation.html
    """
    # Step 1: Create shuffled indices
    # We shuffle to ensure random distribution of classes across folds
    indices = list(range(len(features)))
    random.seed(RANDOM_SEED)  # For reproducibility
    random.shuffle(indices)

    # Step 2: Split indices into n_folds roughly equal parts
    # For example if we have 1000 samples and 5 folds, each fold has ~200 samples
    fold_size = len(indices) // n_folds
    folds = []
    
    for i in range(n_folds):
        start = i * fold_size
        # Last fold gets any remaining samples (handles non-divisible cases)
        end = len(indices) if i == n_folds - 1 else start + fold_size
        folds.append(indices[start:end])

    # Step 3: Perform cross-validation
    fold_accuracies = []

    for fold_idx in range(n_folds):
        # Current fold is used for validation (testing)
        test_indices = folds[fold_idx]

        # All other folds are combined for training
        train_indices = []
        for i in range(n_folds):
            if i != fold_idx:
                train_indices.extend(folds[i])

        # Extract training and validation data using the indices
        train_features = [features[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_features = [features[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]

        # Make predictions on the validation fold
        # verbose=False to reduce output during CV
        predictions = predict_all(train_features, train_labels, test_features,
                                  k_neighbours, verbose=False)

        # Calculate accuracy for this fold
        accuracy = calculate_accuracy(test_labels, predictions)
        fold_accuracies.append(accuracy)

        # Feedback for each fold
        print(f"    Fold {fold_idx + 1}: Accuracy = {accuracy:.4f}")

    # Step 4: Calculate mean and standard deviation
    mean_acc = sum(fold_accuracies) / len(fold_accuracies)
    
    # Standard deviation measures how much fold accuracies vary
    # Low std = consistent performance, High std = unstable
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
    Load feature data from the cleaned CSV file using only standard library.
    
    The CSV file structure (from 02_data_preprocessing.py):
        - Column 0: instance_id (unique identifier)
        - Column 1: filename (original image name)
        - Column 2: folder (which letter folder, A-J)
        - Columns 3-65: 63 features (landmark coordinates)
        - Column 66 (last): label (ASL letter A-J)
    
    We read the features (columns 3-65) and labels (last column),
    ignoring the metadata columns.
    
    Parameters:
        filepath (str): Path to the cleaned CSV file
    
    Returns:
        tuple: (features, labels) where:
            - features: List of lists, each inner list is 63 floats
            - labels: List of strings (A-J)
    """
    features = []
    labels = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header row

        # Features start after: instance_id (0), filename (1), folder (2)
        # So features are columns 3 onwards, excluding the last column (label)
        feature_start = 3
        label_idx = -1  # Last column

        for row in reader:
            # Extract features and convert from strings to floats
            # row[3:-1] gives us columns 3 to second-to-last
            feature_values = [float(x) for x in row[feature_start:label_idx]]
            features.append(feature_values)

            # Extract the label (last column)
            labels.append(row[label_idx])

    return features, labels


# ==============================================================================
# FUNCTION: Standardise Features
# ==============================================================================

def standardise_features(features):
    """
    Standardise features to have mean=0 and standard deviation=1.
    
    This is crucial for kNN because it uses Euclidean distance:
        - Without standardisation, features with larger scales dominate
        - Example: If feature A ranges 0-1000 and feature B ranges 0-1,
          feature A would contribute much more to the distance
    
    The Z-score standardisation formula:
        z = (x - mean) / std
    
    After standardisation:
        - Each feature has mean ≈ 0
        - Each feature has std ≈ 1
        - All features contribute equally to distance calculations
    
    Note: This is equivalent to sklearn's StandardScaler, but implemented
    from scratch using only Python standard library.
    
    Parameters:
        features (list): List of feature vectors (each is a list of floats)
    
    Returns:
        tuple: (standardised_features, means, stds) where:
            - standardised_features: Features after z-score normalisation
            - means: List of mean values for each feature column
            - stds: List of standard deviations for each feature column
            
    The means and stds are returned so they can be applied to test data
    using apply_standardisation().
    
    Reference: https://scikit-learn.org/stable/modules/preprocessing.html
    """
    n_features = len(features[0])  # Number of feature columns (63)
    n_samples = len(features)       # Number of data points

    # Step 1: Calculate mean for each feature column
    means = []
    for j in range(n_features):
        # Sum all values in column j across all samples
        col_sum = sum(features[i][j] for i in range(n_samples))
        means.append(col_sum / n_samples)

    # Step 2: Calculate standard deviation for each feature column
    stds = []
    for j in range(n_features):
        # Variance = average of squared differences from mean
        variance_sum = sum((features[i][j] - means[j]) ** 2 for i in range(n_samples))
        variance = variance_sum / n_samples
        
        # Std = sqrt(variance), but handle zero variance (constant feature)
        # If variance is 0 (all values identical), we use std=1 to avoid division by zero
        stds.append(math.sqrt(variance) if variance > 0 else 1.0)

    # Step 3: Standardise each value using z = (x - mean) / std
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
    Apply standardisation to new data using pre-computed means and stds.
    
    IMPORTANT: When standardising test data, we MUST use the mean and std
    values calculated from the TRAINING data, not from the test data itself.
    
    Why? Because:
    1. In real-world deployment, we won't know test data statistics in advance
    2. Using test statistics would "leak" information from test to training
    3. The model was trained on data with specific scaling; test data must match
    
    This function applies the same transformation that was fitted on training data
    to new data (test set or future predictions).
    
    Parameters:
        features (list): Feature vectors to standardise
        means (list): Pre-computed means from training data
        stds (list): Pre-computed standard deviations from training data
    
    Returns:
        list: Standardised features using the provided means and stds
    """
    standardised = []
    
    for row in features:
        new_row = []
        for j in range(len(row)):
            # Apply same transformation: z = (x - mean) / std
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
    
    We use a random shuffle followed by a split to ensure:
    1. Random distribution of classes in both sets
    2. No ordering bias from the original data
    
    The default 80/20 split is a common standard choice that provides:
    - Enough training data for learning (80%)
    - Enough test data for reliable evaluation (20%)
    
    Reference: Gholamy et al. (2018) 'Why 70/30 or 80/20 Relation Between 
    Training and Testing Sets: A Pedagogical Explanation'
    https://scholarworks.utep.edu/cs_techrep/1209
    
    Note: This is a simple random split. For more rigorous evaluation,
    stratified splitting (maintaining class proportions) is preferred,
    which our sklearn implementation in 04_supervised_learning.py uses.
    
    Parameters:
        features (list): All feature vectors
        labels (list): All corresponding labels
        test_ratio (float): Fraction of data to use for testing (default 0.2)
    
    Returns:
        tuple: (train_features, test_features, train_labels, test_labels)
    """
    # Create indices and shuffle them randomly
    indices = list(range(len(features)))
    random.seed(RANDOM_SEED)  # For reproducibility
    random.shuffle(indices)

    # Calculate where to split (e.g., at 80% for test_ratio=0.2)
    split_idx = int(len(indices) * (1 - test_ratio))

    # Split the shuffled indices
    train_idx = indices[:split_idx]   # First 80%
    test_idx = indices[split_idx:]    # Last 20%

    # Create the actual train and test sets using the indices
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
    Main function - runs the complete kNN from scratch implementation.
    
    This fulfils coursework requirements:
        1. kNN implemented using only Python standard libraries
        2. 5-fold cross-validation for hyperparameter tuning
        3. Testing multiple k values to find optimal setting
        4. Evaluation on held-out test set
        5. Confusion matrix visualisation
    
    Pipeline:
        1. Load cleaned data from CSV
        2. Split into training (80%) and test (20%) sets
        3. Standardise features (fit on training, apply to both)
        4. Use 5-fold CV to find best k value
        5. Evaluate best model on test set
        6. Display results including confusion matrix
    """
    # Pipeline update
    print("=" * 60)
    print("kNN FROM SCRATCH - ASL Hand Pose Classification")
    print("Part 2c: kNN using only Python standard libraries")
    print("=" * 60)

    # Step 0: Check if data file exists error check to catch single pipeline tasks not executed in order.
    if not os.path.exists(DATA_FILE):
        print(f"\nERROR: Data file not found: {DATA_FILE}")
        print("Please run 02_data_preprocessing.py first to generate this file.")
        return

    # ==========================================================================
    # Step 1: Load data from CSV
    # ==========================================================================
    # Pipeline update
    print(f"\nLoading data from: {DATA_FILE}")
    features, labels = load_data_from_csv(DATA_FILE)
    print(f"  Loaded {len(features)} samples with {len(features[0])} features")
    print(f"  Classes: {sorted(set(labels))}")

    # ==========================================================================
    # Step 2: Split into training and test sets
    # ==========================================================================
    # Pipeline update
    print("\nSplitting data (80% train, 20% test)...")
    train_features, test_features, train_labels, test_labels = split_data(
        features, labels, test_ratio=TEST_RATIO
    )
    print(f"  Training samples: {len(train_features)}")
    print(f"  Test samples: {len(test_features)}")

    # ==========================================================================
    # Step 3: Standardise features
    # ==========================================================================
    # Pipeline update
    print("\nStandardising features...")
    # Fit standardisation on training data
    train_std, means, stds = standardise_features(train_features)
    # Apply the same standardisation to test data (using training statistics)
    test_std = apply_standardisation(test_features, means, stds)
    print("  Features standardised (mean≈0, std≈1)")
    print("  Note: Test data standardised using training data statistics")

    # ==========================================================================
    # HYPERPARAMETER TUNING using 5-fold Cross-Validation
    # ==========================================================================
    # Pipeline update
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING (5-fold Cross-Validation)")
    print("=" * 60)
    print(f"\nTesting k values: {K_VALUES}")
    print("(Using odd values to avoid ties in majority voting)")

    results = []
    for k in K_VALUES:
        print(f"\n  Testing k = {k}:")
        cv_result = k_fold_cross_validation(train_std, train_labels, k, 
                                            n_folds=N_FOLDS)
        results.append({
            'k': k,
            'mean_accuracy': cv_result['mean_accuracy'],
            'std_accuracy': cv_result['std_accuracy']
        })
        print(f"    Mean CV Accuracy: {cv_result['mean_accuracy']:.4f} "
              f"(± {cv_result['std_accuracy']:.4f})")

    # Find the best k value (highest mean accuracy)
    best_result = max(results, key=lambda x: x['mean_accuracy'])
    best_k = best_result['k']

    # Pipeline update - Best hyperparameters found
    # Pipeline update
    print("\n" + "-" * 50)
    print("BEST HYPERPARAMETERS FOUND")
    print("-" * 50)
    print(f"  Best k = {best_k}")
    print(f"  CV Accuracy = {best_result['mean_accuracy']:.4f} "
          f"(± {best_result['std_accuracy']:.4f})")

    # ==========================================================================
    # FINAL EVALUATION ON TEST SET
    # ==========================================================================
    # Pipeline update
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    print(f"\n  Making predictions with optimal k = {best_k}...")
    predictions = predict_all(train_std, train_labels, test_std, best_k)

    # Calculate test accuracy
    test_accuracy = calculate_accuracy(test_labels, predictions)
    print(f"\n  TEST SET ACCURACY: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Display confusion matrix
    matrix, labels_list = create_confusion_matrix(test_labels, predictions)
    print_confusion_matrix(matrix, labels_list)

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    # Pipeline update final results
    print("\n" + "=" * 60)
    print("kNN FROM SCRATCH - SUMMARY")
    print("=" * 60)
    print(f"\n  Dataset: ASL Hand Pose Recognition (Letters A-J)")
    print(f"  Features: 63 (21 landmarks × 3 coordinates)")
    print(f"  Training samples: {len(train_std)}")
    print(f"  Test samples: {len(test_std)}")
    print(f"\n  Hyperparameter tuning: 5-fold cross-validation")
    print(f"  k values tested: {K_VALUES}")
    print(f"  Best k value: {best_k}")
    print(f"  Cross-validation accuracy: {best_result['mean_accuracy']:.4f}")
    print(f"  Final test accuracy: {test_accuracy:.4f}")

# ==============================================================================
# RUN THE SCRIPT
# ==============================================================================

if __name__ == "__main__":
    main()
