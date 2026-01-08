"""
================================================================================
main.py - Main Entry Point for ASL Hand Pose Recognition Project
================================================================================

Module:     CMP-6058A/7058A Artificial Intelligence
University: University of East Anglia (UEA)
Author:     [GROUP A108]

This script runs the complete ASL hand pose recognition pipeline:
    1. Feature Extraction (Part 2a) - Extract landmarks from images
    2. Data Preprocessing (Part 2b) - Clean, normalise landmarks, apply SMOTE
    3. kNN from Scratch (Part 2c) - Implement kNN with standard library only
    4. Supervised Learning (Part 2c) - Decision Tree, kNN sklearn,
                                       knn using python libraries only, Random Forest (we wanted to compare knn's
                                       we felt this the best way to check our own versions results)
    5. Unsupervised Learning (Part 2d) - K-Means, Hierarchical clustering

USAGE:  We wanted to be able to test each aspect or run the complete pipeline for ease in development
    python main.py --all              Run complete pipeline
    python main.py --extract          Feature extraction only
    python main.py --preprocess       Preprocessing only
    python main.py --knn              kNN from scratch only
    python main.py --supervised       Supervised learning only
    python main.py --unsupervised     Unsupervised learning only

FOLDER STRUCTURE:
    project/
    ├── data/
    │   └── images/
    │       ├── A/  (ASL letter A images)
    │       ├── B/  (ASL letter B images)
    │       └── ...J/
    ├── output/
    │   ├── asl_features.csv
    │   ├── processed/
    │   └── plots/
    ├── main.py
    ├── 01_feature_extraction.py
    ├── 02_data_preprocessing.py
    ├── 03_knn_from_scratch.py
    ├── 04_supervised_learning.py
    └── 05_unsupervised_learning.py

================================================================================
"""

import argparse
import sys
from datetime import datetime


# ==============================================================================
# CHECK DEPENDENCIES
# ==============================================================================

def check_dependencies():
    """
    Check if all required packages are installed this just made it easy to work on between our group members.

    Returns:
        bool: True if all dependencies are available
    """
    required = [
        ('cv2', 'opencv-python'),
        ('mediapipe', 'mediapipe'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('scipy', 'scipy'),
        ('imblearn', 'imbalanced-learn')  # Added for SMOTE requires python 3.9 to work with mediapipe too!
    ]

    missing = []

    print("\nChecking dependencies...")
    for module_name, package_name in required:
        try:
            __import__(module_name)
            print(f"  [OK] {package_name}")
        except ImportError:
            print(f"  [MISSING] {package_name}")
            missing.append(package_name)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    print("\nAll dependencies OK!")
    return True


# ==============================================================================
# STEP RUNNERS
# ==============================================================================

def run_feature_extraction():
    """Run feature extraction (Part 2a)."""
    #Pipeline update
    print("\n" + "=" * 70)
    print("STEP 1: FEATURE EXTRACTION (Part 2a)")
    print("=" * 70)

    import importlib
    module = importlib.import_module('01_feature_extraction')
    module.main()


def run_preprocessing():
    """Run data preprocessing (Part 2b)."""
    #Pipeline update
    print("\n" + "=" * 70)
    print("STEP 2: DATA PREPROCESSING (Part 2b)")
    print("=" * 70)
    print("Includes: Landmark Normalisation + SMOTE Class Balancing")

    import importlib
    module = importlib.import_module('02_data_preprocessing')
    module.main()


def run_knn_scratch():
    """Run kNN from scratch (Part 2c - Python standard library only)."""
    #Pipeline update
    print("\n" + "=" * 70)
    print("STEP 3: kNN FROM SCRATCH (Part 2c)")
    print("=" * 70)
    print("Using ONLY Python standard libraries (math, csv, collections, random, os)")

    import importlib
    module = importlib.import_module('03_knn_from_scratch')
    module.main()


def run_supervised_learning():
    """Run supervised learning (Part 2c)."""
    #Pipeline update
    print("\n" + "=" * 70)
    print("STEP 4: SUPERVISED LEARNING (Part 2c)")
    print("=" * 70)
    print("Training Decision Tree, kNN (sklearn), and Random Forest")

    import importlib
    module = importlib.import_module('04_supervised_learning')
    module.main()


def run_unsupervised_learning():
    """Run unsupervised learning (Part 2d)."""
    #Pipeline update
    print("\n" + "=" * 70)
    print("STEP 5: UNSUPERVISED LEARNING (Part 2d)")
    print("=" * 70)
    print("Running K-Means and Hierarchical clustering")

    import importlib
    module = importlib.import_module('05_unsupervised_learning')
    module.main()


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """Main entry point."""
    # epilog was a new addition we learnt about here https://docs.python.org/3.9/library/argparse.html#argparse.epilog
    parser = argparse.ArgumentParser(
        description='ASL Hand Pose Recognition Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all              Run complete pipeline
  python main.py --extract          Feature extraction only
  python main.py --preprocess       Preprocessing only
  python main.py --knn              kNN from scratch only
  python main.py --supervised       Supervised learning only
  python main.py --unsupervised     Unsupervised learning only
  python main.py --check            Check dependencies only
        """
    )

    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--extract', action='store_true', help='Feature extraction')
    parser.add_argument('--preprocess', action='store_true', help='Data preprocessing')
    parser.add_argument('--knn', action='store_true', help='kNN from scratch')
    parser.add_argument('--supervised', action='store_true', help='Supervised learning')
    parser.add_argument('--unsupervised', action='store_true', help='Unsupervised learning')
    parser.add_argument('--check', action='store_true', help='Check dependencies')

    args = parser.parse_args()

    # Print header for pipeline
    print("=" * 70)
    print("ASL HAND POSE RECOGNITION PROJECT")
    print("CMP-6058A/7058A Artificial Intelligence - Coursework 2")
    print("University of East Anglia")
    print("-- Group A108 --")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check dependencies as we needed a check to make sure all our dependencies were installed with guidance.
    if not check_dependencies():
        print("\nPlease install missing packages and try again.")
        sys.exit(1)

    if args.check:
        sys.exit(0)

    # Run requested steps for all or individual.
    if args.all:
        run_feature_extraction()
        run_preprocessing()
        run_knn_scratch()
        run_supervised_learning()
        run_unsupervised_learning()
    else:
        if args.extract:
            run_feature_extraction()
        if args.preprocess:
            run_preprocessing()
        if args.knn:
            run_knn_scratch()
        if args.supervised:
            run_supervised_learning()
        if args.unsupervised:
            run_unsupervised_learning()

    # If no options selected, show help here we provided guidance for the user as It's important to give good help
    # for two reasons it firstly shows we know what's going on and required and secondly allows us to negate any
    # potential for building in reproduction errors.
    if not any([args.all, args.extract, args.preprocess, args.knn,
                args.supervised, args.unsupervised, args.check]):
        parser.print_help()
        print("\n" + "-" * 70)
        print("QUICK START")
        print("-" * 70)
        print("""
1. Put your images in data/images/A/, data/images/B/, and so on..

2. Install dependencies (including imbalanced-learn for SMOTE):
   This must be done in a python 3.9 venv due to compatibility with mediapipe and imbalanced-learn and use sub
   version 2 mumpy so numpy<2
   pip install opencv-python mediapipe numpy pandas scikit-learn matplotlib seaborn scipy imbalanced-learn

3. Run the complete pipeline:
   python main.py --all

4. Or run individual steps:
   python main.py --extract      # Step 1: Extract features
   python main.py --preprocess   # Step 2: Clean data + SMOTE
   python main.py --knn          # Step 3: kNN from scratch
   python main.py --supervised   # Step 4: Supervised learning
   python main.py --unsupervised # Step 5: Unsupervised learning

5. Results will be saved in output/plots/
        """)

    print("\n" + "=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
