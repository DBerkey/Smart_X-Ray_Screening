# Smart X-Ray Screening

# Image Patient Analysis

Medical image analysis tool for chest X-ray data visualization and patient statistics.

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure paths in `ImagePatientAnalysis.py`:
   ```python
   store_path = 'path/to/store/analysis'     # Output directory
   source_path = 'path/to/Data_Entry_2017_v2020.csv'  # CSV file
   image_path = 'path/to/images'             # X-ray images folder
   ```

3. Run analysis:
   ```
   python DataAnalysis/ImagePatientAnalysis.py
   ```

## Output
Generates PNG charts analyzing:
- Patient demographics (age/gender)
- Disease patterns and co-occurrence  
- Image characteristics (size, view positions)
- Disease-view correlations

## Requirements
- Python 3.7+
- OpenCV, pandas, matplotlib, seaborn
- Chest X-ray dataset with metadata CSV


# Bbox Dataanalysis

🩻 Chest X-ray Bounding Box Data Analysis

File: Bbox_dataanalysis_updated.ipynb
Dataset: BBox_List_2017.csv

📘 Project Overview

This project performs Exploratory Data Analysis (EDA) on the ChestX-ray14 bounding box dataset.
The dataset provides image filenames, disease labels, and bounding box coordinates (x, y, w, h) marking areas of abnormalities in chest X-rays.

The goal of this notebook is to understand the dataset structure, detect patterns, visualize disease distribution, and analyze bounding box dimensions before building any deep learning or computer vision models.

📂 Dataset Description
Column Name	Description
Image Index	Name of the X-ray image file (e.g., 00013118_008.png)
Finding Label	Disease name or abnormality detected (e.g., Atelectasis)
Bbox [x]	X-coordinate (horizontal start point)
Bbox [y]	Y-coordinate (vertical start point)
Bbox [w]	Bounding box width
Bbox [h]	Bounding box height
⚙️ Setup Instructions

Clone or download the project files into a working directory.

Ensure that the dataset file BBox_List_2017.csv is in the same folder as Bbox_dataanalysis_updated.ipynb.

Install all required dependencies using the command below:

pip install pandas numpy matplotlib seaborn


Open the notebook in Jupyter:

jupyter notebook Bbox_dataanalysis_updated.ipynb


Run all cells sequentially from top to bottom.

🧠 What the Notebook Does
1. Data Loading and Cleaning

Reads the CSV file and displays sample rows.

Checks for null values, duplicate entries, and overall dataset shape.

Converts numeric columns to proper datatypes.

2. Exploratory Data Analysis (EDA)

Counts of unique diseases and their frequencies.

Distribution of bounding box sizes and locations.

Statistical summary (mean, median, std) for bounding box width and height.

3. Data Visualization

Disease frequency bar chart.

Scatter plots showing bounding box positions on a static coordinate scale (so all images are visually consistent).

Histograms of bounding box dimensions.

Area calculation and distribution plots.

4. Filtering and Insights

Filters data for specific diseases (e.g., “Atelectasis”).

Shows relationship between bounding box width, height, and disease type.

Detects any outliers or inconsistencies in bounding box data.

🧩 Code Style

This notebook follows PEP 8 standards:

Clear variable names (e.g., bbox_width, bbox_height)

Proper indentation and line spacing

Inline comments explaining each logical step

Modular cell structure for readability

📊 Outputs

The notebook produces:

Summary statistics tables

Disease distribution bar plots

Scatter plots with consistent axes

Bounding box area histograms

Cleaned dataset preview for modeling
