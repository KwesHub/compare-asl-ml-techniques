"""
================================================================================
01_feature_extraction.py - Feature Extraction from ASL Hand Images
================================================================================

Module:     CMP-6058A/7058A Artificial Intelligence
University: University of East Anglia (UEA)
Author:     [Group A108]

COURSEWORK PART 2a: Download the dataset and extract feature data

We use Google MediaPipe to extract all the hand landmark
coordinates from ASL (American Sign Language) hand pose images we were provided
for this coursework.

WHAT THIS SCRIPT DOES:
    1. Reads images from folders A-J (each folder = one ASL letter which we will call the class)
    2. Uses MediaPipe to detect 21 hand landmarks in each image
    3. Extracts x, y, z coordinates for each landmark (21 × 3 = 63 features)
    4. Saves all features to a CSV file for use in machine learning

FOLDER STRUCTURE REQUIRED as provided:
    data/images/
    ├── A/  (images of ASL letter A)
    ├── B/  (images of ASL letter B)
    and so on...

LEARNING RESOURCES:
    - MediaPipe Hands: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    - OpenCV Python: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
    - Python CSV module: https://docs.python.org/3/library/csv.html

================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
# We need these libraries to read images and detect hand landmarks

import os                      # For working with files and folders
import cv2                     # OpenCV - for reading images
import csv                     # For saving data to CSV files
import mediapipe as mp         # Google's MediaPipe - for hand detection
from datetime import datetime  # For timestamps

# ==============================================================================
# CONFIGURATION - We set them up for our folder
# outputs and image folder inputs to keep everything uniform within our data structure
# ==============================================================================

# Where are the images stored?
IMAGE_FOLDER = "data/images"

# Where should we save the output?
OUTPUT_FOLDER = "output"
OUTPUT_FILE = "output/asl_features.csv"

# MediaPipe settings
# Detection confidence: 0.5 means MediaPipe must be 50% sure it found a hand.
# We will perform some runs with different confidence to see how many images
# MediaPipe will drop or retain. From online research this should be default to
# 0.5 or 50%, but we will raise to 0.7 or 70% for stricter comparative results
# during our testing. At 50% 120 fails, at 70% 176 fails.
# That equates to 46.7% more failures at 0.7 than 0.5 so MediaPipe can give us
# different datasets to test later in the project for comparison.
DETECTION_CONFIDENCE = 0.5

# We only expect one hand per image
MAX_HANDS = 1

# For this coursework the images are all .jpg, but we thought to add
# provision for more types here, useful in a situation with mixed filetypes.
IMAGE_TYPES = ('.jpg', '.jpeg')

# The ASL letters we're working with (A to J = 10 classes)
# In regard to classifying a major step as we are basically
# telling the system what hand gesture it should be.
ASL_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']


# ==============================================================================
# FUNCTION: Set up MediaPipe hand detector
# ==============================================================================

def create_hand_detector():
    """
    First we needed to create and configure the MediaPipe hand detector.

    MediaPipe Hands can detect 21 landmarks on a hand:
        - Wrist (1 point)
        - Thumb (4 points)
        - Index finger (4 points)
        - Middle finger (4 points)
        - Ring finger (4 points)
        - Pinky (4 points)

    Returns:
        hands: The configured hand detector object which simply saves
               any confusion and allows variable changes in one location.
    """
    # Access MediaPipe's hand detection solution
    mp_hands = mp.solutions.hands

    # Create the detector with our settings
    # static_image_mode=True because we're processing single images, not video
    # but interestingly from our research live video would be possible for
    # live detection which would solve the confusion with I and J.
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=DETECTION_CONFIDENCE
    )

    return hands


# ==============================================================================
# FUNCTION: Extract landmarks from one image
# ==============================================================================

def extract_landmarks(image_path, detector):
    """
    Extract the 21 hand landmark coordinates from a single image.

    Each landmark has 3 values:
        - x: horizontal position (0 to 1, left to right)
        - y: vertical position (0 to 1, top to bottom)
        - z: depth/distance from camera (relative value)

    Total features: 21 landmarks × 3 coordinates = 63 features
    as explained in the brief.

    We will be providing the path and detector values.

    Parameters:
        image_path: Path to the image file
        detector: The MediaPipe hand detector

    Returns:
        list: 63 float values, or None if no hand was detected.
              Important for keeping failure counts.
    """
    # Step 1: Read the image using OpenCV
    image = cv2.imread(image_path)

    # Check if image was loaded successfully
    if image is None:
        return None

    # Step 2: Convert from BGR to RGB colour format
    # OpenCV uses BGR, but MediaPipe expects RGB.
    # This is an important step.
    # Reference: https://medium.com/the-digit/unveiling-the-art-of-opencv-and-mediapipe-project-polygonboundary-456824dc1dc4
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 3: Process the image to detect hands
    results = detector.process(image_rgb)

    # Step 4: Check if a hand was found
    if results.multi_hand_landmarks is None:
        # No hand detected - this counts as "noise" for clarification for the coursework
        return None #so we can still keep a count

    # Step 5: Get the first hand's landmarks
    hand = results.multi_hand_landmarks[0]

    # Step 6: Extract x, y, z for all 21 landmarks
    # Reference: https://mediapipe.readthedocs.io/en/latest/solutions/hands.html#static_image_mode
    features = []
    for landmark in hand.landmark:
        features.append(landmark.x)  # x coordinate
        features.append(landmark.y)  # y coordinate
        features.append(landmark.z)  # z coordinate (depth)

    # We should have exactly 63 values (21 × 3)
    return features


# ==============================================================================
# FUNCTION: Process all images and save to CSV
# ==============================================================================

def process_all_images():
    """
    Process all images in the dataset and save features to CSV.

    The CSV file will have these columns:
        instance_id: Unique number for each image
        filename: Original image filename
        folder: Which letter folder (A-J)
        63 feature columns (landmark_0_x, landmark_0_y, landmark_0_z, etc.)
        label: The ASL letter (A-J)
    """
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Create the hand detector
    detector = create_hand_detector()

    # Keep track of statistics for our failure rates
    total_images = 0
    successful = 0
    failed = 0

    # Create column names for the CSV
    # First 3 columns: instance_id, filename, folder
    columns = ['instance_id', 'filename', 'folder']

    # Add 63 feature columns (one for each coordinate)
    for i in range(21):
        columns.append(f'landmark_{i}_x')
        columns.append(f'landmark_{i}_y')
        columns.append(f'landmark_{i}_z')

    # Last column: the label (ASL letter)
    columns.append('label')

    # Open CSV file for writing
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)  # Write header row first

        instance_id = 1

        # Go through each letter folder (A-J)
        for letter in ASL_LETTERS:
            folder_path = os.path.join(IMAGE_FOLDER, letter)

            # Skip if folder doesn't exist to prevent breaking the process
            if not os.path.exists(folder_path):
                print(f"  Warning: Folder '{letter}' not found, skipping this one...")
                continue

            # Get all image files in this folder using our image type variable,
            # in effect allowing us to use any images we specify
            images = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(IMAGE_TYPES)]

            folder_success = 0
            folder_fail = 0

            # User feedback to show its progress
            print(f"  Processing folder '{letter}' ({len(images)} images)...")

            # Process each image
            for filename in images:
                total_images += 1  # Required for totals
                image_path = os.path.join(folder_path, filename)

                # Extract landmarks
                features = extract_landmarks(image_path, detector)

                if features is None:
                    # Hand detection failed
                    failed += 1  # Noise None required for totals
                    folder_fail += 1
                    continue

                if len(features) != 63:
                    # Wrong number of features (shouldn't happen but we needed to be prepared for it)
                    failed += 1  # Noise Invalid required for totals
                    folder_fail += 1
                    continue

                # Write row to CSV
                row = [instance_id, filename, letter] + features + [letter]
                writer.writerow(row)

                instance_id += 1
                successful += 1
                folder_success += 1

            # Feedback
            print(f" {folder_success} successful, {folder_fail} failed")

    # Close detector
    detector.close()

    # Return statistics
    return {
        'total': total_images,
        'successful': successful,
        'failed': failed
    }


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Main function - runs when we execute this script.

    We realised we needed options and error control in regard to
    folder names which we made easy by making IMAGE_FOLDER an
    editable variable. We wanted clear feedback on the MediaPipe
    extraction process when testing.
    """
    print("=" * 60)
    print("FEATURE EXTRACTION - ASL Hand Pose Recognition")
    print("Part 2a: Extract feature data from images")
    print("=" * 60)
    print(f"\nStart time: {datetime.now().strftime('%H:%M:%S')}")

    # Check if image folder exists. If it is not found we thought
    # it would be useful to advise of the structure expected.
    # This follows the structure of the extracted images we received.
    # we felt providing instruction on error would help.
    if not os.path.exists(IMAGE_FOLDER):
        print(f"\nERROR: Image folder not found: {IMAGE_FOLDER}")
        print("\nPlease create this folder structure:")
        print("  data/images/A/  (put A images here)")
        print("  data/images/B/  (put B images here)")
        print("...    continue for all other Letters")
        return

    # Process all images
    print(f"\nProcessing images from: {IMAGE_FOLDER}")
    print("-" * 60)

    stats = process_all_images()

    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\n  Total images processed: {stats['total']}")     # Image totals
    print(f"  Successful extractions: {stats['successful']}")  # Success totals
    print(f"  Failed (no hand found): {stats['failed']}")      # Failed totals

    # Nice to add a percentage score on success rate, useful for a
    # method graph in our academic poster later
    if stats['total'] > 0:
        success_rate = (stats['successful'] / stats['total']) * 100
        print(f"  Success rate: {success_rate:.1f}%")

    # Confirming the output save location and end time
    print(f"\n  Output saved to: {OUTPUT_FILE}")
    print(f"\nEnd time: {datetime.now().strftime('%H:%M:%S')}")


# ==============================================================================
# RUN THE SCRIPT
# ==============================================================================

if __name__ == "__main__":
    main()