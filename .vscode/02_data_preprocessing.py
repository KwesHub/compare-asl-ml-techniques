"""
================================================================================
02_data_preprocessing.py - Data Preprocessing and Cleaning
================================================================================

Module:     CMP-6058A/7058A Artificial Intelligence
University: University of East Anglia (UEA)
Author:     [Group A108]

COURSEWORK PART 2b: Pre-process data

This script prepares the extracted feature data for machine learning by:
    1. Loading the feature data from CSV
    2. Removing duplicates and handling missing values
    3. Removing noise/outliers (failed detections)
    4. Applying Hand Landmark Normalisation (position/scale invariance)
    5. Splitting data into training (80%) and test (20%) sets
    6. Applying SMOTE for class balancing - we added this after researching
       importance of SMOTE for training data bias as we felt there was a large
       gap between some totals
    7. Normalising features using StandardScaler
    8. Creating visualisations of the data (before and after)

================================================================================
PREPROCESSING TECHNIQUES APPLIED
================================================================================

1. HAND LANDMARK NORMALISATION
   ---------------------------
   Raw MediaPipe coordinates are absolute positions within the image frame.
   This means the same hand gesture photographed at different positions would
   have different feature values, which could confuse the classifier.

   We apply two normalisation steps:
   a) Position Invariance: Subtract wrist (landmark 0) coordinates from all
      landmarks, making the wrist the origin (0,0,0). This removes the effect
      of where the hand appears in the image.

   b) Scale Invariance: Divide all coordinates by the palm width (distance
      from wrist to middle finger MCP). This normalises for different hand
      sizes and distances from the camera.

   Reference:
       Zhang, F. et al. (2020) 'MediaPipe Hands: On-device Real-time Hand
       Tracking', arXiv preprint. Available at: https://arxiv.org/abs/2006.10214

       Google Developers (2024) 'Hand landmarks detection guide', MediaPipe.
       Available at: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

2. SMOTE (Synthetic Minority Over-sampling Technique)
   ---------------------------------------------------
   Class imbalance can bias classifiers towards majority classes, reducing
   accuracy for minority classes. Our dataset has slight imbalance (e.g.,
   class C has 302 samples while class F has 414 samples - a 37% difference).

   SMOTE addresses this by generating synthetic samples for minority classes
   through interpolation between existing samples and their k-nearest neighbours.
   This is preferred over random oversampling (which just duplicates samples)
   because it creates new, diverse samples that help the model generalise.

   IMPORTANT: SMOTE is applied ONLY to training data AFTER the train/test split.
   Applying it before the split would cause data leakage, where synthetic samples
   based on test data contaminate the training set.

   Reference:
       Chawla, N.V. et al. (2002) 'SMOTE: Synthetic Minority Over-sampling
       Technique', Journal of Artificial Intelligence Research, 16, pp. 321-357.
       Available at: https://arxiv.org/abs/1106.1813

       Imbalanced-learn Documentation (2024) 'SMOTE'. Available at:
       https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

================================================================================
WHY THESE PREPROCESSING STEPS MATTER
================================================================================

Machine learning pipelines require careful data preparation to:
- Ensure FAIR comparison between classifiers (all see the same quality of data)
- Reduce BIAS from class imbalance that could skew results
- Improve GENERALISATION by making features position/scale invariant we could provide different image sizes and
  distances and still get reasonable data.
- Enable REPRODUCIBILITY through consistent preprocessing

These steps follow best practices from the machine learning literature and lectures
we have had. This includes lectures on algorithmic bias and AI in our attempt to
demonstrate understanding of data quality issues in real-world applications.

It is important to note we had to downgrade our Python version to 3.9 to
run MediaPipe and imbalanced-learn as this was the only version that both can support
contrary to the course recommendations. However, we realise this was an extra
step to add SMOTE and data bias consideration to our testing. Therefore we have
taken the required steps to make it work with mediapipe.

================================================================================
LEARNING RESOURCES
================================================================================
Scikit-learn preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
Imbalanced-learn: https://imbalanced-learn.org/stable/
Train/test split: https://scikit-learn.org/stable/modules/cross_validation.html
Pandas documentation: https://pandas.pydata.org/docs/
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import pandas as pd                          # For data manipulation
import numpy as np                           # For numerical operations
import matplotlib.pyplot as plt              # For creating plots
import seaborn as sns                        # For nice statistical plots
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import StandardScaler      # For normalising features
from sklearn.preprocessing import LabelEncoder        # For encoding labels
import os
from datetime import datetime

# SMOTE for class balancing
# Reference: Chawla et al. (2002) - https://arxiv.org/abs/1106.1813
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    #feedback for user if not setup correctly
    print("WARNING: imbalanced-learn not installed. Run: pip install imbalanced-learn "
          "MUST use Python 3.9 for imbalanced-learn & MediaPipe")
    print("         SMOTE class balancing will be skipped.")

# Use non-interactive backend for saving plots (we learnt this prevents display issues)
import matplotlib
matplotlib.use('Agg')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Input file (from our feature extraction)
INPUT_FILE = "output/asl_features.csv"

# Output directories for processed files and our comparison plots
OUTPUT_DIR = "output/processed"
PLOTS_DIR = "output/plots"

# The ASL letters we expect (A to J). This can easily be edited to include
# the entire alphabet and more for expansion later, a best practice for expansion we considered.
VALID_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# Train/Test split ratio
# 80% for training, 20% for testing (common standard in ML)
# Gholamy, Afshin; Kreinovich, Vladik; and Kosheleva, Olga, "Why 70/30 or 80/20
# Relation Between Training and Testing Sets: A Pedagogical Explanation" (2018).
# Departmental Technical Reports (CS). 1209.
# Available at: https://scholarworks.utep.edu/cs_techrep/1209
TEST_SIZE = 0.2

# Random seed for reproducibility
# Using the same seed means we get the same split every time,
# so this must be random to reproduce the result without the
# same starting seed to help.
RANDOM_STATE = 42

# Landmark indices for normalisation
# Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
WRIST_INDEX = 0           # Landmark 0 is the wrist
MIDDLE_MCP_INDEX = 9      # Landmark 9 is the middle finger MCP (base)


# ==============================================================================
# FUNCTION: Load the feature data
# ==============================================================================

def load_data(filepath):
    """
    Load the extracted feature data from CSV file.

    Parameters:
        filepath: Path to the CSV file

    Returns:
        DataFrame containing the feature data
    """
    # Making sure we have plenty of process feedback for
    # usability and testing
    print(f"  Loading data from: {filepath}")

    # Read CSV using pandas
    df = pd.read_csv(filepath)

    print(f"  Loaded {len(df)} samples with {len(df.columns)} columns")

    # Dataframe returned
    return df


# ==============================================================================
# FUNCTION: Explore the data
# ==============================================================================

def explore_data(df):
    """
    Print useful information about the dataset.

    This helps us understand:
        - How many samples we have per class or "letter"
        - If there are any missing values
        - If there are any duplicates
    """
    print("\n" + "-" * 50)
    print("DATA EXPLORATION")
    print("-" * 50)

    # Basic shape
    print(f"\n  Total samples: {len(df)}")          # Sample count
    print(f"  Total columns: {len(df.columns)}")    # Column count
    print(f"  Features: 63 (21 landmarks × 3 coordinates)")  # Remind of totals per image

    # Class distribution - how many samples per ASL letter?
    # Makes sense to show a simple count and percentage display
    # for each
    print("\n  Samples per class:")
    class_counts = df['label'].value_counts().sort_index()
    for label, count in class_counts.items():
        percentage = (count / len(df)) * 100
        bar = '\u2588' * int(count / 50)  # Simple visual bar
        print(f"    {label}: {count:4d} ({percentage:5.1f}%) {bar}")

    # Check for missing values by creating a dataframe of true/false for each value.
    # .isnull() then .sum() counts `True` values in each column,
    # then second .sum() adds up all column counts into one total.
    missing = df.isnull().sum().sum()
    print(f"\n  Missing values: {missing}")

    # Check for duplicate instance IDs so we check the column exists then
    # select just the instance_id and .duplicated() applies a boolean
    # for each row and then .sum() totals the True duplicate count.
    if 'instance_id' in df.columns:
        duplicates = df['instance_id'].duplicated().sum()
        print(f"  Duplicate IDs: {duplicates}")

    # Calculate class imbalance ratio
    max_count = class_counts.max()  # Highest sample count
    min_count = class_counts.min()  # Lowest sample count
    imbalance_ratio = max_count / min_count  # Ratio of both
    print(f"\n  Class imbalance ratio: {imbalance_ratio:.2f}:1 (max/min)")
    print(f"  Largest class: {class_counts.idxmax()} ({max_count} samples)")   # Label with most samples
    print(f"  Smallest class: {class_counts.idxmin()} ({min_count} samples)")  # Label with fewest samples


# ==============================================================================
# FUNCTION: Clean the data
# ==============================================================================

def clean_data(df):
    """
    Clean the dataset by removing any problems.

    What we clean:
        1. Remove rows with missing values (incomplete data)
        2. Remove duplicate instance IDs (repeated entries)
        3. Keep only valid labels (A-J)
    """
    #Pipeline update
    print("\n" + "-" * 50)
    print("DATA CLEANING")
    print("-" * 50)

    original_count = len(df)

    # Step 1: Remove rows with any missing values
    # Reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
    df_clean = df.dropna()
    after_missing = len(df_clean)  # Count remaining rows
    removed_missing = original_count - after_missing  # Show how many removed
    print(f"\n  Removed {removed_missing} rows with missing values")

    # Step 2: Remove duplicate instance IDs (keep first occurrence)
    if 'instance_id' in df_clean.columns:
        # Reference: https://medium.com/@aniketbakre1291/removing-duplicates-from-datasets-keeping-the-first-or-last-row-2a97da12dd6b
        # Checking for duplicates in the column and only keeping the first.
        df_clean = df_clean.drop_duplicates(subset=['instance_id'], keep='first')
        after_duplicates = len(df_clean)
        removed_duplicates = after_missing - after_duplicates
        print(f"  Removed {removed_duplicates} duplicate entries")

    # Step 3: Keep only valid labels (A-J)
    # Count how many we have BEFORE
    before_label_filter = len(df_clean)
    # Keep only rows where label is in our valid list (A-J).
    # .isin() returns boolean for each row.
    df_clean = df_clean[df_clean['label'].isin(VALID_LABELS)]
    # Count how many we have AFTER filtering
    after_labels = len(df_clean)
    # How many were removed
    removed_labels = before_label_filter - after_labels
    print(f"  Removed {removed_labels} rows with invalid labels")

    # Summary totals for feedback and testing
    total_removed = original_count - len(df_clean)
    print(f"\n  Original samples: {original_count}")
    print(f"  After cleaning:   {len(df_clean)}")
    print(f"  Total removed:    {total_removed} ({(total_removed/original_count)*100:.1f}%)")

    # Return our cleaned data
    return df_clean


# ==============================================================================
# FUNCTION: Normalise hand landmarks (Position and Scale Invariance)
# ==============================================================================

def normalise_hand_landmarks(df):
    """
    Apply position and scale normalisation to hand landmark coordinates.

    This preprocessing step ensures that hand gestures are recognised regardless
    of where the hand appears in the image or how far it is from the camera, which
    we feel is critical based on the knowledge that the dataset is of many students
    and various hand sizes and shapes.

    Normalisation Steps:
        1. Position Invariance: Subtract wrist coordinates from all landmarks
           - Makes wrist the origin (0, 0, 0)
           - Removes effect of hand position in image

        2. Scale Invariance: Divide by palm width (wrist to middle finger MCP distance)
           - Normalises for different hand sizes
           - Normalises for different distances from camera

    Reference:
        Zhang, F. et al. (2020) 'MediaPipe Hands: On-device Real-time Hand Tracking'
        Available at: https://arxiv.org/abs/2006.10214

        Google Developers (2024) 'Hand landmarks detection guide'
        Available at: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    """
    print("\n" + "-" * 50)
    print("HAND LANDMARK NORMALISATION")
    print("-" * 50)
    print("\n  Applying position and scale invariance...")

    # Create a copy to avoid modifying original important for our testing checks
    df_norm = df.copy()

    # Get wrist coordinates (landmark 0) for position normalisation
    wrist_x = df_norm['landmark_0_x'].values
    wrist_y = df_norm['landmark_0_y'].values
    wrist_z = df_norm['landmark_0_z'].values

    # Get middle finger MCP coordinates (landmark 9) for scale calculation
    mcp_x = df_norm['landmark_9_x'].values
    mcp_y = df_norm['landmark_9_y'].values
    mcp_z = df_norm['landmark_9_z'].values

    # Calculate palm width (Euclidean distance from wrist to middle MCP)
    # This serves as a scale factor for normalisation
    # distance = sqrt((x2−x1)² + (y2−y1)² + (z2−z1)²)
    palm_width = np.sqrt(
        (mcp_x - wrist_x)**2 +
        (mcp_y - wrist_y)**2 +
        (mcp_z - wrist_z)**2
    )

    # Avoid division by zero (use small epsilon for any zero values).
    # np.where is like an if/else for arrays, checking if value is zero, negative or tiny
    # positive numbers vs just <= 0 for catching extra edge cases.
    # Then we replace it with 0.000001 because dividing by zero causes infinity
    # errors.
    palm_width = np.where(palm_width < 1e-6, 1e-6, palm_width)

    # Apply normalisation to all 21 landmarks
    for i in range(21):
        # Position normalisation: subtract wrist coordinates
        # Scale normalisation: divide by palm width which should make
        # all hands the same size

        x_col = f'landmark_{i}_x'
        y_col = f'landmark_{i}_y'
        z_col = f'landmark_{i}_z'

        # Subtract wrist position (position invariance)
        df_norm[x_col] = df_norm[x_col] - wrist_x
        df_norm[y_col] = df_norm[y_col] - wrist_y
        df_norm[z_col] = df_norm[z_col] - wrist_z

        # Divide by palm width (scale invariance)
        df_norm[x_col] = df_norm[x_col] / palm_width
        df_norm[y_col] = df_norm[y_col] / palm_width
        df_norm[z_col] = df_norm[z_col] / palm_width

    # Use our same feedback style and testing display
    print("\n  Position invariance: Wrist set as origin (0, 0, 0)")
    print("  Scale invariance: Coordinates normalised by palm width")
    print(f"  Palm width stats: mean={np.mean(palm_width):.4f}, std={np.std(palm_width):.4f}")

    # Return our normalised dataframe
    return df_norm


# ==============================================================================
# FUNCTION: Create visualisations
# ==============================================================================

def create_visualisations(df, suffix="", is_normalised=False):
    """
    Create helpful visualisations of the data.

    We need to create:
        1. Class/Letter distribution bar chart
        2. Feature correlation heatmap
        these will serve us in our presentation and poster
        for comparison and showing how skewed the original dataset is pre SMOTE

    Parameters:
        df: DataFrame with feature data
        suffix: String to add to plot titles (e.g., " (Before)" or " (After Normalisation)")
        is_normalised: If True, skip landmark_0 in correlation plot (since all zeros)
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Class Distribution Bar Chart
    plt.figure(figsize=(10, 6))
    class_counts = df['label'].value_counts().sort_index()

    # Use different colours for before/after comparison
    colour = 'steelblue' if 'Before' in suffix else 'forestgreen'

    bars = plt.bar(class_counts.index, class_counts.values, color=colour, alpha=0.8)

    # Add count labels on top of each bar
    for bar, count in zip(bars, class_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(count), ha='center', fontsize=10)

    plt.xlabel('ASL Letter', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f'Class Distribution{suffix}', fontsize=14)
    plt.tight_layout()

    filename = f'class_distribution{suffix.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150) #higher dpi for poster may be needed but will test at 150
    plt.close()
    print(f"  Saved: {filename}")

    # 2. Feature Correlation Heatmap
    # Get feature columns
    feature_cols = [col for col in df.columns
                    if col not in ['instance_id', 'filename', 'folder', 'label']]

    # After normalisation, skip landmark_0 (all zeros = no correlation possible)
    if is_normalised:
        # Start from landmark_1 (skip first 3 columns which are landmark_0_x/y/z)
        sample_features = feature_cols[3:15]  # landmark_1 to landmark_4 (12 features)
        title_note = "\n(Landmark 0 excluded - all zeros after normalisation)"
    else:
        # Before normalisation, include landmark_0
        sample_features = feature_cols[:12]  # landmark_0 to landmark_3 (12 features)
        title_note = ""

    plt.figure(figsize=(10, 8))
    correlation = df[sample_features].corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True)
    plt.title(f'Feature Correlations (12 Features){suffix}{title_note}', fontsize=12)
    plt.tight_layout()

    filename = f'feature_correlations{suffix.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ==============================================================================
# FUNCTION: Create Normalisation Verification Plot
# ==============================================================================

def create_normalisation_verification_plot(df_before, df_after):
    """
    Create a visualisation proving that landmark normalisation is working.

    After normalisation:
        - landmark_0_x, landmark_0_y, landmark_0_z should ALL be 0
        - This proves we successfully made the wrist the origin (0, 0, 0)

    This is useful for the academic poster to demonstrate understanding
    of the preprocessing step.

    Parameters:
        df_before: DataFrame BEFORE normalisation (cleaned data)
        df_after: DataFrame AFTER normalisation
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Landmark 0 (Wrist) Normalisation Verification\n'
                 'Position Invariance: Wrist becomes origin (0, 0, 0)',
                 fontsize=14, fontweight='bold')

    # Landmark 0 coordinates to check
    coords = ['landmark_0_x', 'landmark_0_y', 'landmark_0_z']
    coord_names = ['X Coordinate', 'Y Coordinate', 'Z Coordinate']

    # Top row: BEFORE normalisation (values vary)
    for i, (col, name) in enumerate(zip(coords, coord_names)):
        ax = axes[0, i]
        values = df_before[col].values

        ax.hist(values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'{name} Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'BEFORE: {col}', fontsize=11, fontweight='bold')

        # Add statistics box
        stats_text = f'Mean: {np.mean(values):.4f}\nStd: {np.std(values):.4f}\nMin: {np.min(values):.4f}\nMax: {np.max(values):.4f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Bottom row: AFTER normalisation (all zeros)
    for i, (col, name) in enumerate(zip(coords, coord_names)):
        ax = axes[1, i]
        values = df_after[col].values

        # Check if all values are zero (or very close to zero)
        all_zero = np.allclose(values, 0, atol=1e-10)

        # Create histogram - but since all zeros, we just show a single bar to highlight the success
        if all_zero:
            ax.bar([0], [len(values)], width=0.1, color='forestgreen', alpha=0.7, edgecolor='black')
            ax.set_xlim(-0.5, 0.5)
            ax.set_title(f'AFTER: {col}\n ALL ZEROS', fontsize=11, fontweight='bold', color='green')
        else:
            ax.hist(values, bins=50, color='red', alpha=0.7, edgecolor='black')
            ax.set_title(f'AFTER: {col}\n NOT ZERO', fontsize=11, fontweight='bold', color='red')

        ax.set_xlabel(f'{name} Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)

        # Add statistics box
        stats_text = f'Mean: {np.mean(values):.2e}\nStd: {np.std(values):.2e}\nAll Zero: {all_zero}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen' if all_zero else 'lightcoral', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'normalisation_verification.png'), dpi=150)
    plt.close()
    print(f"  Saved: normalisation_verification.png")

    # Print verification to console
    print("\n  NORMALISATION VERIFICATION:")
    for col in coords:
        before_std = np.std(df_before[col].values)
        after_mean = np.mean(df_after[col].values)
        after_std = np.std(df_after[col].values)
        all_zero = np.allclose(df_after[col].values, 0, atol=1e-10)

        print(f"    {col}:")
        print(f"      Before: std = {before_std:.6f} (values vary)")
        print(f"      After:  mean = {after_mean:.2e}, std = {after_std:.2e}")
        if all_zero:
            print(f"       VERIFIED: All values are zero")
        else:
            print(f"       WARNING: Values are not all zero")


# ==============================================================================
# FUNCTION: Apply SMOTE for class balancing
# ==============================================================================

def apply_smote(X_train, y_train, label_encoder):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.

    We were concerned seeing all the image totals after MediaPipe that because
    we don't start with a fair number it is hard to do a fair and equal training data test.
    Class imbalance can bias classifiers towards majority classes. SMOTE addresses
    this by generating synthetic samples for minority classes through interpolation
    between existing samples and their k-nearest neighbours.

    IMPORTANT: SMOTE is applied ONLY to training data to prevent data leakage.
    This is also the main reason we needed to use Python 3.9 as both MediaPipe
    and Imbalanced-learn will only work together in this version of Python
    interpreter.

    Reference:
        Chawla, N.V. et al. (2002) 'SMOTE: Synthetic Minority Over-sampling
        Technique', Journal of Artificial Intelligence Research, 16, pp. 321-357.
        Available at: https://arxiv.org/abs/1106.1813

        Imbalanced-learn Documentation (2024) 'SMOTE'
        Available at: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

    Parameters:
        X_train: Training features
        y_train: Training labels (encoded)
        label_encoder: The LabelEncoder used for labels

    Returns:
        tuple: (X_train_balanced, y_train_balanced)
    """
    #Pipeline update progress
    print("\n" + "-" * 50)
    print("SMOTE CLASS BALANCING")
    print("-" * 50)

    # Some simple error checking and handling
    if not SMOTE_AVAILABLE:
        print("\n  WARNING: imbalanced-learn not installed, skipping SMOTE")
        print("  Install with: pip install imbalanced-learn, must be Python version 3.9 ONLY.")
        return X_train, y_train

    # Show class distribution before SMOTE
    print("\n  Class distribution BEFORE SMOTE:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        label_name = label_encoder.inverse_transform([u])[0]
        print(f"    {label_name}: {c:4d} samples")

    # Apply SMOTE
    # k_neighbors=5 is the default and commonly used value
    # Reference: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

    # Show class distribution after SMOTE giving us a clear before and after for feedback and later plots
    print("\n  Class distribution AFTER SMOTE:")
    unique, counts = np.unique(y_balanced, return_counts=True)
    for u, c in zip(unique, counts):
        label_name = label_encoder.inverse_transform([u])[0]
        print(f"    {label_name}: {c:4d} samples")

    # Again using our feedback and testing display to verify the process
    print(f"\n  Original training samples: {len(X_train)}")
    print(f"  Balanced training samples: {len(X_balanced)}")
    print(f"  Synthetic samples added: {len(X_balanced) - len(X_train)}")
    print("\n  NOTE: SMOTE applied only to training data (not test) to prevent data leakage")

    return X_balanced, y_balanced


# ==============================================================================
# FUNCTION: Prepare data for machine learning
# ==============================================================================

def prepare_for_ml(df):
    """
    Prepare the cleaned data for machine learning algorithms.

    Steps:
        1. Separate features (X) from labels (y)
        2. Encode labels as numbers (A=0, B=1, etc.)
        3. Split into training and test sets
        4. Apply SMOTE to balance training classes
        5. Normalise features using StandardScaler

    Why StandardScaler?
        - It transforms features to have mean=0 and std=1
        - This is important for kNN because it uses distances
        - Without scaling, features with larger values would dominate

    Parameters:
        df: The cleaned DataFrame

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, label_encoder)
    """
    #Pipeline update progress
    print("\n" + "-" * 50)
    print("PREPARING DATA FOR ML")
    print("-" * 50)

    # Get feature column names (everything except metadata and label)
    feature_cols = [col for col in df.columns
                   if col not in ['instance_id', 'filename', 'folder', 'label']]

    print(f"\n  Number of features: {len(feature_cols)}")

    # Separate features (X) and labels (y)
    X = df[feature_cols].values  # Convert to numpy array
    y = df['label'].values

    # Encode labels: A=0, B=1, C=2, etc.
    # ML algorithms need numbers, not letters!
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Feedback and test print of conversion
    print(f"\n  Label encoding:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"    {label} → {i}")

    # Split into training and test sets.
    # stratify=y_encoded ensures equal class proportions in both sets. To explain:
    # Random split might put 90% of class C in training and only 10% in test giving
    # unreliable evaluation for that class. This introduces bias.

    # We justify a 20/80 split based on knowledge learnt from:
    # Gholamy, A., Kreinovich, V. and Kosheleva, O. (2018) 'Why 70/30 or 80/20
    # Relation Between Training and Testing Sets: A Pedagogical Explanation',
    # Departmental Technical Reports (CS), 1209.
    # Available at: https://scholarworks.utep.edu/cs_techrep/1209

    # Important observations from research:
    # Split   Training Data   Test Data                   Trade-off
    # 90/10   More learning   Less reliable evaluation    Risk: test set too small
    # 80/20   Good balance    Reliable evaluation         Sweet spot
    # 70/30   Less learning   Very reliable evaluation    Risk: not enough training but acceptable
    # 50/50   Too little      Overkill for testing        Model underfits

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,  # We set this to 0.2 = 20% for testing
        random_state=RANDOM_STATE,
        stratify=y_encoded  # IMPORTANT: stratify will maintain class balance
    )

    # Feedback and data split information for testing
    print(f"\n  Data split ({int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test):")
    print(f"    Training samples: {len(X_train)}")
    print(f"    Test samples:     {len(X_test)}")

    # Apply SMOTE to balance training classes
    # IMPORTANT: Only apply to training data, never to test data
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train, label_encoder)

    # Normalise features using StandardScaler
    # fit_transform on training data, then transform test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)  # Learn from training data
    X_test_scaled = scaler.transform(X_test)                  # Apply to test data
    #some user feedback
    print(f"\n  Applied StandardScaler normalisation")
    print(f"    Features now have mean ≈ 0 and std ≈ 1")

    return X_train_scaled, X_test_scaled, y_train_balanced, y_test, feature_cols, label_encoder


# ==============================================================================
# FUNCTION: Save processed data
# ==============================================================================

def save_processed_data(X_train, X_test, y_train, y_test, feature_names, label_encoder):
    """
    Save the processed data to files for use by other scripts,
    as running preprocessing once, then reusing it many times increases speed.

    Here is our usage through our pipeline:
        01_feature_extraction.py    - Extracts features from images
        02_data_preprocessing.py    - Cleans & prepares data  - SAVES FILES HERE
        03_knn_from_scratch.py      - Loads saved files - USES SAVED FILES
        04_supervised_learning.py   - Loads saved files - USES SAVED FILES
        05_unsupervised_learning.py - Loads saved files - USES SAVED FILES

    We save:
        - X_train.npy, X_test.npy: Feature arrays
        - y_train.npy, y_test.npy: Label arrays
        - feature_names.txt: List of feature column names
        - label_mapping.txt: Mapping of numbers to letters

    File descriptions:
        X_train.npy     Training features (3,310 × 63 array)    Training classifiers
        X_test.npy      Test features (733 × 63 array)          Evaluating classifiers
        y_train.npy     Training labels (3,310 numbers)         Training classifiers
        y_test.npy      Test labels (733 numbers)               Evaluating classifiers
        feature_names.txt   Column names like landmark_0_x      Labelling plots/charts
        label_mapping.txt   0,A 1,B 2,C etc.                    Converting predictions back to letters

    Parameters:
        All the processed data and metadata
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save numpy arrays
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)

    # Save feature names
    with open(os.path.join(OUTPUT_DIR, 'feature_names.txt'), 'w') as f:
        for name in feature_names:
            f.write(name + '\n')

    # Save label mapping (number -> letter)
    with open(os.path.join(OUTPUT_DIR, 'label_mapping.txt'), 'w') as f:
        for i, label in enumerate(label_encoder.classes_):
            f.write(f"{i},{label}\n")

    # Feedback for testing and verification
    print(f"\n  Data saved to: {OUTPUT_DIR}")
    print("    - X_train.npy, X_test.npy (features)")
    print("    - y_train.npy, y_test.npy (labels)")
    print("    - feature_names.txt")
    print("    - label_mapping.txt")


# ==============================================================================
# FUNCTION: Create before/after comparison for SMOTE
# ==============================================================================

def create_smote_comparison_plot(y_train_original, y_train_balanced, label_encoder):
    """
    Create a comparison plot showing class distribution before and after SMOTE.
    We felt this was important for showing our results and proving the now balanced training dataset.

    Args:
        y_train_original: Training labels before SMOTE
        y_train_balanced: Training labels after SMOTE
        label_encoder: LabelEncoder for converting numbers to letters
    """
    # Make sure our plot output folder exists and don't crash if it does for repeat
    # attempts
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 1 row, 2 columns of charts
                                                      # Figure size: 14 inches wide, 5 inches tall

    # Before SMOTE axes 0
    unique_before, counts_before = np.unique(y_train_original, return_counts=True)  # Array of unique before labels & counts
    # Convert the numbers back to letters as before using imbalanced learn we had to convert to numbers.
    labels_before = [label_encoder.inverse_transform([u])[0] for u in unique_before]
    axes[0].bar(labels_before, counts_before, color='steelblue', alpha=0.8)  # Bar chart left, X-axis labels (A, B, C...)
                                                                              # Bar heights (280, 327, 242...), blue, trans
    for i, (label, count) in enumerate(zip(labels_before, counts_before)):  # Pairs up labels with counts: ('A', 280)
        axes[0].text(i, count + 5, str(count), ha='center', fontsize=9)  # Adds number label above each bar
    axes[0].set_xlabel('ASL Letter', fontsize=11)
    axes[0].set_ylabel('Number of Samples', fontsize=11)
    axes[0].set_title('Training Data BEFORE SMOTE', fontsize=12)
    axes[0].set_ylim(0, max(counts_before) * 1.15)  # Y-axis from 0 to 115% of tallest bar (leaves room for number labels)

    # After SMOTE axes 1 - only difference is right chart and using balanced data and different colour
    unique_after, counts_after = np.unique(y_train_balanced, return_counts=True)
    labels_after = [label_encoder.inverse_transform([u])[0] for u in unique_after]
    axes[1].bar(labels_after, counts_after, color='forestgreen', alpha=0.8)
    for i, (label, count) in enumerate(zip(labels_after, counts_after)):
        axes[1].text(i, count + 5, str(count), ha='center', fontsize=9)
    axes[1].set_xlabel('ASL Letter', fontsize=11)
    axes[1].set_ylabel('Number of Samples', fontsize=11)
    axes[1].set_title('Training Data AFTER SMOTE (Balanced)', fontsize=12)
    axes[1].set_ylim(0, max(counts_after) * 1.15)

    plt.suptitle('SMOTE Class Balancing Effect', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'smote_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: smote_comparison.png")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Main function - runs the complete preprocessing pipeline.
    """
    #Pipeline update
    print("=" * 60)
    print("DATA PREPROCESSING - ASL Hand Pose Recognition")
    print("Part 2b: Clean and prepare data")
    print("=" * 60)

    # Start time for measuring time taken on different systems we added this out of interest for high end systems and
    # basic systems.
    print(f"\nStart time: {datetime.now().strftime('%H:%M:%S')}")

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"\nERROR: Feature file not found: {INPUT_FILE}")
        print("Please run 01_feature_extraction.py first.")
        return

    # Step 1: Load the data
    df = load_data(INPUT_FILE)

    # Check for empty dataset
    if len(df) == 0:
        print("\nERROR: Dataset is empty!")
        return

    # Step 2: Explore the data
    explore_data(df)

    # Step 3: Create visualisations (before cleaning)
    print("\n" + "-" * 50)
    print("CREATING VISUALISATIONS (Before Preprocessing)")
    print("-" * 50)
    create_visualisations(df, suffix=" (Before)", is_normalised=False)

    # Step 4: Clean the data (remove missing values, duplicates)
    df_clean = clean_data(df)

    # Check if cleaning removed everything - error checking
    if len(df_clean) == 0:
        print("\nERROR: All data was removed during cleaning!")
        return

    # Step 5: Apply Hand Landmark Normalisation
    # Reference: Zhang et al. (2020) - https://arxiv.org/abs/2006.10214
    df_normalised = normalise_hand_landmarks(df_clean)

    # Step 6: Save cleaned & normalised CSV for reference
    cleaned_csv = os.path.join(OUTPUT_DIR, 'asl_features_cleaned.csv')
    df_normalised.to_csv(cleaned_csv, index=False)
    print(f"\n  Cleaned & normalised CSV saved to: {cleaned_csv}")
    #Pipeline update
    print("\n" + "-" * 50)
    print("NORMALISATION VERIFICATION")
    print("-" * 50)
    create_normalisation_verification_plot(df_clean, df_normalised)

    # Step 7: Create visualisations (after landmark normalisation)
    print("\n" + "-" * 50)
    print("CREATING VISUALISATIONS (After Landmark Normalisation)")
    print("-" * 50)
    create_visualisations(df_normalised, suffix=" (After Normalisation)", is_normalised=True)

    # Step 8: Prepare data for machine learning (includes SMOTE)
    X_train, X_test, y_train, y_test, features, encoder = prepare_for_ml(df_normalised)

    # Step 9: Create SMOTE comparison visualisation
    # Need to get original training labels before SMOTE for comparison
    #Pipeline update
    print("\n" + "-" * 50)
    print("CREATING SMOTE COMPARISON VISUALISATION")
    print("-" * 50)

    # Re-split to get original training labels (before SMOTE) for visualisation
    X_temp = df_normalised[[col for col in df_normalised.columns
                            if col not in ['instance_id', 'filename', 'folder', 'label']]].values
    y_temp = encoder.transform(df_normalised['label'].values)
    _, _, y_train_original, _ = train_test_split(
        X_temp, y_temp, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_temp
    )
    create_smote_comparison_plot(y_train_original, y_train, encoder)

    # Step 10: Save processed data
    save_processed_data(X_train, X_test, y_train, y_test, features, encoder)

    #Pipeline update COMPLETE!
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print("\n  PREPROCESSING STEPS APPLIED:")
    print("  1. Data cleaning (missing values, duplicates)")
    print("  2. Hand landmark normalisation")
    print("  3. Train/test split (80/20 with stratification)")
    print("  4. SMOTE class balancing (training data only)")
    print("  5. StandardScaler normalisation")

    # End time to complete our system measurement
    print(f"\nEnd time: {datetime.now().strftime('%H:%M:%S')}")


# ==============================================================================
# RUN THE SCRIPT
# ==============================================================================

if __name__ == "__main__":
    main()