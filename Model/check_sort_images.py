"""
Author: Douwe Berkeij
Date: 13-10-2025
"""

import os
import pandas as pd

def check_images_present(images_folder_path, df):
    """
    Param:
        images_folder_path: str : path to the folder containing images
        df: pd.DataFrame : dataframe containing metadata with 'Image Index' column
    Returns:
        bool : True if all images are present, else raises ValueError
    """
    file_names = df['Image Index'].unique()
    expected_count = len(file_names)

    actual_count = 0    
    # List all files in the images folder
    files = os.listdir(images_folder_path)

    for file_name in file_names:
        if file_name in files:
            actual_count += 1
      
    if actual_count == expected_count:
        return True
    else:
        raise ValueError(f"Expected {expected_count} images, but found {actual_count} images in the folder.")

def split_p_sex_data(DATA_DIRECTORY_PATH, images_folder_path, df):
    """
    Param:
        DATA_DIRECTORY_PATH: str : path to the main data directory
        images_folder_path: str : path to the folder containing images
        df: pd.DataFrame : dataframe containing metadata with 'Image Index' and 'Patient Sex' columns
    Returns:
        None : splits images into folders based on patient sex
    """
    sex_present = df['Patient Sex'].unique()

    # Create directories if they don't exist
    for sex in sex_present:
        dir_path = os.path.join(DATA_DIRECTORY_PATH, f'images_{sex}')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    # Move files to corresponding directories
    for index, row in df.iterrows():
        file_name = row['Image Index']
        patient_sex = row['Patient Sex']
        src_path = os.path.join(images_folder_path, file_name)
        dest_path = os.path.join(DATA_DIRECTORY_PATH, f'images_{patient_sex}', file_name)
        if os.path.exists(src_path):
            os.rename(src_path, dest_path)
        else:
            print(f"File {file_name} not found in {images_folder_path}")
            print(f"Moving {file_name} to {dest_path}")
            os.rename(src_path, dest_path)

if __name__ == "__main__":
    DATA_DIRECTORY_PATH = "path/to/your/data"
    images_folder_path = DATA_DIRECTORY_PATH + "/images"
    images_metadata_path = DATA_DIRECTORY_PATH + "/Data_Entry_2017_v2020.csv"

    df = pd.read_csv(images_metadata_path)

    if not (check_images_present(images_folder_path, df)):
        raise ValueError("Not all images are present in the folder.")

    split_p_sex_data(DATA_DIRECTORY_PATH, images_folder_path, df)
