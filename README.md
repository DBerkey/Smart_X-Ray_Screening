# Smart X-Ray Screening
The Smart X-Ray Screening tool is an result of the IKT213 course at the UiA.
# Data Analyses 
The Data analyses of the CHESTXRAY14 dataset is split in two parts, an analyses of the `Data_Entry_2017_v2020.csv` and one of the `BBox_List_2017.csv`. The first analyses focuses on the pations in the dataset in relation to the abnormalitys present and the second on the location of the abnormalitys in the images.
## Image Patient Analysis

Medical image analysis tool for data visualization and patient statistics of the `Data_Entry_2017_v2020.csv` file.

### Setup
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

### Output
Generates PNG charts analyzing:
- Patient demographics (age/gender)
- Disease patterns and co-occurrence  
- Image characteristics (size, view positions)
- Disease-view correlations

### Requirements
- Python 3.7+
- OpenCV, pandas, matplotlib, seaborn
- Chest X-ray dataset with metadata CSV


## Condition location analysis

Analyzes the spatial distribution and characteristics of medical conditions in chest X-ray images using bounding box data from the `BBox_List_2017.csv` dataset.

### Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure path in `Bbox_dataanalysis.ipynb`:
   ```python
   DATA_PATH = "path/to/BBox_List_2017.csv"
   ```

3. Run analysis:
   ```bash
   jupyter notebook DataAnalysis/Bbox_dataanalysis.ipynb
   ```
### Outputs

- Summary statistics tables
- Disease distribution bar plots
- Scatter plots with consistent axes
- Bounding box area histograms
- Cleaned dataset preview for modeling

### Requirements
- Python 3.7+
- pandas, matplotlib, seaborn, jupyter
- Chest X-ray dataset with BBox_List_2017.csv file

