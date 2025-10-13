"""
Author: Douwe Berkeij
Date: 13-10-2025
AI use: in this code there was made use of GitHub Copilot to generate the docstrings and debuging
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
import pandas as pd
import os
import numpy as np
from PIL import Image

def split_data_train_test_eval(x, y, test_size=0.195, eval_size=0.005): 
    """
    Splits the data into training, testing, and evaluation sets.
    
    Param:
        x: Features
        y: Labels
        test_size: Proportion of the dataset to include in the test split.
        eval_size: Proportion of the dataset to include in the evaluation split.
    Returns:
        (x_train, y_train), (x_test, y_test), (x_eval, y_eval): Tuples containing the split data.
    """
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=test_size + eval_size)
    x_test, x_eval, y_test, y_eval = train_test_split(x_temp, y_temp, test_size=eval_size / (test_size + eval_size))
    return (x_train, y_train), (x_test, y_test), (x_eval, y_eval)

def load_image_as_vector(image_path, img_size):
    """
    Loads an image, resizes it, converts to grayscale, and flattens it into a vector.
    Param:
        image_path: Path to the image file.
        img_size: Tuple specifying the desired image size (width, height).
    Returns:
        Flattened image as a numpy array.
    """
    img = Image.open(image_path).convert("L")
    img = img.resize(img_size)
    return np.array(img).flatten()

def build_feature_matrix(images_folder_path, df, img_size):
    """
    Builds a feature matrix from images specified in the dataframe.
    Param:
        images_folder_path: Path to the folder containing images.
        df: DataFrame containing image metadata.
        img_size: Tuple specifying the desired image size (width, height).
    Returns:
        Numpy array where each row is a flattened image.
    """
    X_features = []
    for img_name in df["Image Index"]:
        path = os.path.join(images_folder_path, img_name)
        X_features.append(load_image_as_vector(path, img_size))
    return np.array(X_features)

def encode_predictor_labels(y_series):
    """
    Encodes the labels using MultiLabelBinarizer.
    Param:
        y_series: Series containing the labels.
    Returns:
        Encoded labels and the MultiLabelBinarizer instance.
    """
    y_list = [labels.split('|') for labels in y_series]
    mlb = MultiLabelBinarizer()
    Y_encoded = mlb.fit_transform(y_list)
    return Y_encoded, mlb

def encode_scale_input_features(df, X_img):
    """
    Encodes and scales non-image features
    Param:
        df: DataFrame containing the features.
        X_img: Numpy array of image features.
    Returns:
        Combined feature matrix with image and non-image features.
    """
    age = df[['Patient Age']].values
    age_scaled = StandardScaler().fit_transform(age)
    
    sex_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    sex_encoded = sex_enc.fit_transform(df[['Patient Sex']])
    
    X_combined = np.hstack([X_img, age_scaled, sex_encoded])
    return X_combined

if __name__ == "__main__":
    DATA_DIRECTORY_PATH = "C:/Users/berke/OneDrive/Documenten/school/UiA/Smart_X-Ray_Screening_img-data"
    images_folder_path = DATA_DIRECTORY_PATH + "/images_M"
    images_metadata_path = DATA_DIRECTORY_PATH + "/Data_Entry_2017_v2020.csv"

    df = pd.read_csv(images_metadata_path)

    # Filter the metadata to only include images present in the folder
    image_names = set(os.listdir(images_folder_path))
    df = df[df['Image Index'].isin(image_names)]

    x = df[['Image Index', 'Patient Age', 'Patient Sex']]
    y = df['Finding Labels']

    train, test, eval = split_data_train_test_eval(x, y)

    # Save evaluation data to CSV for application use
    eval_csv_path = os.path.join(DATA_DIRECTORY_PATH, "eval_data.csv")
    eval_df = pd.DataFrame(eval[0])
    eval_df['Finding Labels'] = eval[1].values
    eval_df.to_csv(eval_csv_path, index=False)
    print(f"Evaluation data saved to {eval_csv_path}")

    # Build feature matrices
    img_size = (2500, 2048) # !Important! still need to get the real size of the images
    X_train = build_feature_matrix(images_folder_path, pd.DataFrame(train[0]), img_size)
    X_test = build_feature_matrix(images_folder_path, pd.DataFrame(test[0]), img_size)

    Y_train, mlb = encode_predictor_labels(train[1])
    Y_test, _ = encode_predictor_labels(test[1])

    X_train_encoded = encode_scale_input_features(pd.DataFrame(train[0]), X_train)
    X_test_encoded = encode_scale_input_features(pd.DataFrame(test[0]), X_test)

    print(f"Classes: {mlb.classes_}")
    print(f"Number of classes: {len(mlb.classes_)}")
    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of testing samples: {X_test.shape[0]}")
    print(f"Feature vector size: {X_train.shape[1]}")
    print(f"Image vector size: {img_size[0]*img_size[1]}")
