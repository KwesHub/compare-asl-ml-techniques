ASL Hand Pose Recognition 

Project Setup Guide 

PyCharm & Spyder/Anaconda on Windows 

CMP-6058A/7058A Artificial Intelligence Coursework 2 

Group A108 

University of East Anglia  

Contents 

1. Required Files 

2. Required Folder Structure 

3. Required Python Packages 

4. Setup Option A: PyCharm 

5. Setup Option B: Spyder with Anaconda 

6. Running the Project 

7. Troubleshooting Common Issues 

 

1. Required Files 

The following Python files are required to run the complete ASL Hand Pose Recognition pipeline: 

File Name 

Purpose 

main.py 

Main entry point - orchestrates the entire pipeline 

01_feature_extraction.py 

Extracts hand landmarks using MediaPipe (Part 2a) 

02_data_preprocessing.py 

Cleans data, applies SMOTE & normalisation (Part 2b) 

03_knn_from_scratch.py 

kNN using only Python standard libraries (Part 2c) 

04_supervised_learning.py 

Decision Tree, kNN, Random Forest classifiers (Part 2c) 

05_unsupervised_learning.py 

K-Means and Hierarchical clustering (Part 2d) 

06_visualization.py 

Utility functions for poster visualisations (Part 2e) 

2. Required Folder Structure 

Create the following folder structure on your computer. The exact structure is important for the scripts to find the image data and save outputs correctly.  We used Pycharm to setup the interpreter and would recommend it over more convoluted systems like Anaconda or Spyder. 

Complete Folder Structure: 

ProjectFolder/ 

│ 

├── main.py 

├── 01_feature_extraction.py 

├── 02_data_preprocessing.py 

├── 03_knn_from_scratch.py 

├── 04_supervised_learning.py 

├── 05_unsupervised_learning.py 

├── 06_visualization.py 

│ 

├── data/ 

│   └── images/ 

│       ├── A/          ← Put ASL letter A images here 

│       ├── B/          ← Put ASL letter B images here 

│       ├── C/          ← Put ASL letter C images here 

│       ├── D/          ← Put ASL letter D images here 

│       ├── E/          ← Put ASL letter E images here 

│       ├── F/          ← Put ASL letter F images here 

│       ├── G/          ← Put ASL letter G images here 

│       ├── H/          ← Put ASL letter H images here 

│       ├── I/          ← Put ASL letter I images here 

│       └── J/          ← Put ASL letter J images here 

│ 

└── output/             ← Created automatically by scripts 

    ├── asl_features.csv 

    ├── processed/ 

    └── plots/ 

Important Notes: 

The folder names A, B, C, etc. must be uppercase letters 

Image files should be .jpg, .jpeg, .png, or .bmp format 

The output/ folder will be created automatically when running the scripts 

All Python files must be in the same folder (not in subfolders) 

 

3. Required Python Packages 

The following Python packages must be installed. Python 3.9 only!  We had to make sure our virtual environments were in Python version 3.9 to get both imbalanced-learn and mediapipe installed, it is also important to note that numpy must be a sub version 2 install. 

Package 

Purpose 

Install Command 

opencv-python 

Image reading and processing 

pip install opencv-python 

mediapipe 

Hand landmark detection 

pip install mediapipe 

numpy 

Numerical computing 

pip install numpy<2 

pandas 

Data manipulation 

pip install pandas 

scikit-learn 

ML algorithms and tools 

pip install scikit-learn 

matplotlib 

Creating plots and charts 

pip install matplotlib 

seaborn 

Statistical visualisations 

pip install seaborn 

scipy 

Scientific computing 

pip install scipy 

imbalanced-learn 

SMOTE class balancing 

pip install imbalanced-learn 

Quick Install - All Packages in One Command: 

pip install opencv-python mediapipe numpy<2 pandas scikit-learn matplotlib seaborn scipy imbalanced-learn 

 

4. Setup Option A: PyCharm 

PyCharm is a professional Python IDE. Follow these steps to set up the project in PyCharm on Windows. 

4.1 Install PyCharm 

Download PyCharm Community Edition (free) from: https://www.jetbrains.com/pycharm/download/ 

Run the installer and follow the prompts 

Launch PyCharm after installation 

4.2 Create a New Project 

Click 'New Project' on the welcome screen 

Set the Location to your desired project folder (e.g., C:\Users\YourName\ASL_Project) 

Under 'Python Interpreter', select 'New environment using Virtualenv' 

Make sure 'Base interpreter' is set to Python 3.8 or higher 

Click 'Create' 

4.3 Add Project Files 

Copy all Python files (main.py, 01_feature_extraction.py, etc.) into the project folder 

Create the data/images/ folder structure inside the project folder 

Copy the anonymised dataset images into the appropriate A, B, C... J folders 

In PyCharm, right-click the project folder and select 'Reload from Disk' to see the files 

4.4 Install Required Packages 

Open Terminal in PyCharm (View → Tool Windows → Terminal) 

Run the following command: 

pip install opencv-python mediapipe numpy pandas scikit-learn matplotlib seaborn scipy imbalanced-learn 

Wait for all packages to install (this may take a few minutes) 

Verify installation by running: python main.py --check 

4.5 Run the Project 

Open main.py in the editor 

Click the green 'Run' button (or right-click → Run 'main') 

To run with arguments, go to Run → Edit Configurations 

Add --all to the 'Parameters' field to run the complete pipeline 

 

5. Setup Option B: Spyder with Anaconda 

Anaconda is a Python distribution that includes Spyder IDE and many scientific packages. This is often used in university labs. 

5.1 Install Anaconda 

Download Anaconda from: https://www.anaconda.com/download 

Run the installer (choose 'Just Me' and accept default options) 

Important: Tick 'Add Anaconda to PATH' if prompted (or remember to use Anaconda Prompt) 

5.2 Create a Conda Environment 

Open Anaconda Prompt (search for it in the Start menu) and run these commands: 

# Create a new environment called 'asl_project' with Python 3.9 

conda create -n asl_project python=3.9 

# Activate the environment 

conda activate asl_project 

# Install required packages 

pip install opencv-python mediapipe numpy pandas scikit-learn matplotlib seaborn scipy imbalanced-learn 

# Install Spyder in this environment 

conda install spyder 

5.3 Set Up Project Folder 

Create a project folder (e.g., C:\Users\YourName\ASL_Project) 

Copy all Python files into this folder 

Create the data/images/ folder structure 

Copy images into the A, B, C... J folders 

5.4 Launch Spyder 

In Anaconda Prompt (with environment activated): 

# Make sure environment is activated 

conda activate asl_project 

# Navigate to project folder 

cd C:\Users\YourName\ASL_Project 

# Launch Spyder 

spyder 

5.5 Configure Spyder Working Directory 

In Spyder, go to Tools → Preferences 

Click 'Current working directory' in the left panel 

Set 'At startup, the current working directory is' to your project folder 

Click OK 

5.6 Run the Project in Spyder 

Open main.py in Spyder (File → Open) 

To run the complete pipeline, you have two options: 

Option A - Modify the script: 

Add the following line at the bottom of main.py before running: 

sys.argv = ['main.py', '--all'] 

Option B - Use Spyder's IPython console: 

Type in the console: 

%run main.py --all 

 

6. Running the Project 

6.1 Command Line Options 

The main.py script accepts the following command line arguments: 

Command 

Description 

python main.py --all 

Run the complete pipeline (all 5 steps) 

python main.py --check 

Check if all dependencies are installed 

python main.py --extract 

Run only feature extraction (Step 1) 

python main.py --preprocess 

Run only preprocessing with SMOTE (Step 2) 

python main.py --knn 

Run only kNN from scratch (Step 3) 

python main.py --supervised 

Run only supervised learning (Step 4) 

python main.py --unsupervised 

Run only unsupervised learning (Step 5) 

6.2 Expected Output Files 

After running the complete pipeline, you will find these files in the output folder: 

output/asl_features.csv - Raw extracted features 

output/processed/asl_features_cleaned.csv - Cleaned and normalised features 

output/processed/*.npy - Training and test data arrays 

output/plots/*.png - All visualisation plots for the poster 

 

7. Troubleshooting Common Issues 

7.1 ModuleNotFoundError 

Error: ModuleNotFoundError: No module named 'mediapipe' 

Solution: The package is not installed. Run the pip install command again. Make sure you're using the correct Python environment. 

7.2 imbalanced-learn Not Found 

Error: ModuleNotFoundError: No module named 'imblearn' 

Solution: Run: pip install imbalanced-learn (note: the package name is different from the import name) 

7.3 Image Folder Not Found 

Error: ERROR: Image folder not found: data/images 

Solution: Create the data/images/ folder structure and add image subfolders A through J. Make sure the current working directory is set to the project folder. 

7.4 Working Directory Issues 

Error: Files not found even though they exist 

Solution: The working directory may be wrong. In Spyder, check Tools → Preferences → Current working directory. In PyCharm, check Run → Edit Configurations → Working directory. 

You can also add this to the start of main.py to set the directory: 

import os 

os.chdir(r'C:\Users\YourName\ASL_Project') 

7.5 MediaPipe Warnings 

Warning: TensorFlow Lite XNNPACK delegate warnings appear 

Solution: These warnings are normal and can be ignored. MediaPipe uses TensorFlow Lite internally and these are informational messages, not errors. 

7.6 Slow Performance 

Issue: The kNN from scratch step is very slow 

Explanation: This is expected. The from-scratch kNN implementation uses only Python standard libraries (no NumPy), so it's much slower than the scikit-learn version. This is required by the coursework to demonstrate understanding of the algorithm. 

7.7 Memory Errors 

Error: MemoryError during processing 

Solution: Close other applications to free up memory. If using a laptop, make sure it's plugged in for maximum performance. Alternatively, you can run individual steps instead of --all to process in smaller chunks. 