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


# Image Preprocessing

### Prerequisites
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your data:
   - Create an `images/` directory in the `Preprocessing/` folder
   - Place your X-ray images (.png, .jpg, .jpeg, .bmp, .tiff) in the `images/` directory

### Running Preprocessing

1. Navigate to Preprocessing directory
```bash
cd Preprocessing
```

2. Run batch processing (processes all images)
```python
python preprocessing.py
```
This will:
- Process all images in the `images/` directory
- Save enhanced images to `preprocessed/` directory
- Apply soft tissue optimization for better radiologic visibility

3. Process single image with pipeline visualization
```python
# Edit the 'input_of_single_image' in preprocessing.py 
# OR run in Python:
from preprocessing import preprocess_xray_soft_tissue
processed_image = preprocess_xray_soft_tissue('images/your_image.png', show_process=True)
```
This creates a `soft_tissue_pipeline/` folder showing each processing step.

4. Run tests (optional)
```bash
python test_preprocessing.py
```

### Output
- Enhanced X-ray images optimized for soft tissue detection
- Pipeline visualization showing processing steps
- Batch processing progress tracking

# Training Machine learning Models
### Prerequisites
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your data and directories:
   - Ensure your image data is available in a folder (e.g., `images-preprocessed`)
   - Place the metadata CSV file (e.g., `Data_Entry_2017_v2020.csv`) in your data directory
   - Update the paths in `train_model_twostage.py`:
     ```python
     DATA_DIRECTORY_PATH = "path/to/your/data-directory"
     images_folder_path = DATA_DIRECTORY_PATH + "/images-preprocessed"
     images_metadata_path = DATA_DIRECTORY_PATH + "/Data_Entry_2017_v2020.csv"
     ```

### Running the training

   ```bash
   python Model/train_model_twostage.py
   ```

   - set `LOAD_EXISTING_DATA` flag for if you want to reuse previous proccesd data
   - Set `TRAIN_STAGE1` and `TRAIN_STAGE2` flags in the script to control which models are trained or loaded.
   - Preprocessed features and trained models will be saved in the data directory.

### Output
- Trained binary and multi-label classifiers
- Saved model files (`.pkl`)
- Feacher scalers (`.pkl`)
- Preprocessed feature files (`.npz`)
- Console summary of training and evaluation metrics

# Webaplication
### Prerequisites
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your data and directories:
   - to come

### Run Flask
```bash
$env:FLASK_APP = "WebAplication/flaskr"
python -m flask run
```