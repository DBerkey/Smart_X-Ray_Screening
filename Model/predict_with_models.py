"""
Author: Douwe Berkeij
Date: 09-11-2025
Script to make predictions using trained models from train_model_twostage.py
Preprocessing matches the training pipeline.
"""

import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog

MODEL_DIR = os.path.dirname(__file__)
TRAINED_MODELS_DIR = os.path.join(MODEL_DIR, 'trained_models')
IMG_SIZE = (256, 256)
SCALER_PATH = os.path.join(TRAINED_MODELS_DIR, 'feature_scaler.pkl')


def extract_hog_features(image_path, img_size, pixels_per_cell=(32, 32), cells_per_block=(4, 4), orientations=9):
    """
    param:
    image_path: str, path to the image file
    img_size: tuple, desired image size (width, height)
    returns:
    HOG features as a numpy array
    """
    img = Image.open(image_path).convert("L").resize(img_size)
    img_np = np.array(img) / 255.0
    features = hog(img_np, pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block, orientations=orientations, feature_vector=True)
    return features.reshape(1, -1)


def encode_scale_input_features(df, X_img, scaler, age_scaler, sex_enc, view_enc):
    """
    param:
    df: DataFrame, containing metadata
    X_img: numpy array, image features
    scaler: StandardScaler, fitted scaler for image features
    age_scaler: StandardScaler, fitted
    sex_enc: OneHotEncoder, fitted
    view_enc: OneHotEncoder, fitted
    returns:
    Combined and scaled features as a numpy array
    """
    # Scale image features using the loaded scaler
    img_scaled = scaler.transform(X_img)
    # Encode and scale metadata using loaded encoders
    age = df[['Patient Age']].values
    age_scaled = age_scaler.transform(age)
    sex_encoded = sex_enc.transform(df[['Patient Sex']])
    view_encoded = view_enc.transform(df[['View Position']])
    X_combined = np.hstack([img_scaled, age_scaled, sex_encoded, view_encoded])
    return X_combined


def load_model(model_path):
    """
    param:
    model_path: str, path to the saved model file
    returns:
    Loaded model
    """
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_scaler(scaler_path):
    """
    param:
    scaler_path: str, path to the saved scaler file
    returns:
    Loaded scaler
    """
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def preprocess_input(image_path, patient_age, patient_sex, view_position, scaler, age_scaler, sex_enc, view_enc):
    """
    param:
    image_path: str, path to the image file
    patient_age: int, age of the patient
    patient_sex: str, sex of the patient
    view_position: str, view position of the image
    returns:
    Preprocessed and encoded features as a numpy array
    """
    # Build DataFrame for metadata
    df = pd.DataFrame({
        'Patient Age': [patient_age],
        'Patient Sex': [patient_sex],
        'View Position': [view_position]
    })
    # HOG image features
    x_img = extract_hog_features(image_path, IMG_SIZE)
    # Combine features using loaded encoders
    x_encoded = encode_scale_input_features(
        df, x_img, scaler, age_scaler, sex_enc, view_enc)
    return x_encoded


def predict(image_path, patient_age, patient_sex, view_position,
            stage1_model_path, stage2_model_paths, scaler_path=SCALER_PATH):
    """
    param:
    image_path: str, path to the image file
    patient_age: int, age of the patient
    patient_sex: str, sex of the patient
    view_position: str, view position of the image
    stage1_model_path: str, path to the stage 1 model
    stage2_model_paths: dict, mapping condition labels to their model paths
    scaler_path: str, path to the feature scaler
    returns:
    finding_pred: int, prediction from stage 1 (0 or 1)
    results: dict or None, predictions from stage 2 if finding_pred is 1
    """
    scaler = load_scaler(scaler_path)
    # Load encoders
    age_scaler_path = os.path.join(TRAINED_MODELS_DIR, 'age_scaler.pkl')
    sex_encoder_path = os.path.join(TRAINED_MODELS_DIR, 'sex_encoder.pkl')
    view_encoder_path = os.path.join(TRAINED_MODELS_DIR, 'view_encoder.pkl')
    with open(age_scaler_path, 'rb') as f:
        age_scaler = pickle.load(f)
    with open(sex_encoder_path, 'rb') as f:
        sex_enc = pickle.load(f)
    with open(view_encoder_path, 'rb') as f:
        view_enc = pickle.load(f)
    x_encoded = preprocess_input(
        image_path, patient_age, patient_sex, view_position, scaler, age_scaler, sex_enc, view_enc)
    stage1_model = load_model(stage1_model_path)
    finding_pred = stage1_model.predict(x_encoded)[0]
    print(f"Stage 1 (Finding/No Finding): {finding_pred}")
    if finding_pred == 1:
        results = {}
        for label, model_path in stage2_model_paths.items():
            model = load_model(model_path)
            pred = model.predict(x_encoded)[0]
            results[label] = pred
        print("Stage 2 (Conditions):", results)
        return finding_pred, results
    else:
        print("No findings detected.")
        return finding_pred, None


if __name__ == "__main__":
    # --- CONFIG ---
    stage1_model_path = os.path.join(MODEL_DIR, '../svm_stage1_binary.pkl')
    stage2_model_paths = {
        "Atelectasis": os.path.join(MODEL_DIR, '../svm_stage2_Atelectasis.pkl'),
        "Cardiomegaly": os.path.join(MODEL_DIR, '../svm_stage2_Cardiomegaly.pkl'),
        "Consolidation": os.path.join(MODEL_DIR, '../svm_stage2_Consolidation.pkl'),
        "Edema": os.path.join(MODEL_DIR, '../svm_stage2_Edema.pkl'),
        "Effusion": os.path.join(MODEL_DIR, '../svm_stage2_Effusion.pkl'),
        "Emphysema": os.path.join(MODEL_DIR, '../svm_stage2_Emphysema.pkl'),
        "Fibrosis": os.path.join(MODEL_DIR, '../svm_stage2_Fibrosis.pkl'),
        "Hernia": os.path.join(MODEL_DIR, '../svm_stage2_Hernia.pkl'),
        "Infiltration": os.path.join(MODEL_DIR, '../svm_stage2_Infiltration.pkl'),
        "Mass": os.path.join(MODEL_DIR, '../svm_stage2_Mass.pkl'),
        "Nodule": os.path.join(MODEL_DIR, '../svm_stage2_Nodule.pkl'),
        "Pleural_Thickening": os.path.join(MODEL_DIR, '../svm_stage2_Pleural_Thickening.pkl'),
        "Pneumonia": os.path.join(MODEL_DIR, '../svm_stage2_Pneumonia.pkl'),
        "Pneumothorax": os.path.join(MODEL_DIR, '../svm_stage2_Pneumothorax.pkl'),
    }
    image_path = "path_to_image.jpg"  # Change to your image
    patient_age = 45  # Example
    patient_sex = "Male"  # Example
    view_position = "PA"  # Example
    predict(image_path, patient_age, patient_sex, view_position,
            stage1_model_path, stage2_model_paths)
