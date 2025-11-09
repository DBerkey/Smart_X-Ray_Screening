
"""
Script to make predictions using trained models from train_model_twostage.py
Preprocessing matches the training pipeline.
"""

import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from skimage.feature import hog

def extract_hog_features(image_path, img_size, pixels_per_cell=(32, 32), cells_per_block=(4, 4), orientations=9):
    img = Image.open(image_path).convert("L").resize(img_size)
    img_np = np.array(img) / 255.0
    features = hog(img_np, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, orientations=orientations, feature_vector=True)
    return features.reshape(1, -1)


def encode_scale_input_features(df, X_img, scaler=None):
    age = df[['Patient Age']].values
    age_scaled = StandardScaler().fit_transform(age) if scaler is None else scaler.transform(age)
    sex_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    sex_encoded = sex_enc.fit_transform(df[['Patient Sex']])
    view_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    view_encoded = view_enc.fit_transform(df[['View Position']])
    X_combined = np.hstack([X_img, age_scaled, sex_encoded, view_encoded])
    return X_combined

def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def load_scaler(scaler_path):
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def preprocess_input(image_path, patient_age, patient_sex, view_position, scaler):
    # Build DataFrame for metadata
    df = pd.DataFrame({
        'Patient Age': [patient_age],
        'Patient Sex': [patient_sex],
        'View Position': [view_position]
    })
    # HOG image features
    x_img = extract_hog_features(image_path, IMG_SIZE)
    # Combine features
    x_encoded = encode_scale_input_features(df, x_img, scaler)
    return x_encoded

def predict(image_path, patient_age, patient_sex, view_position,
            stage1_model_path, stage2_model_paths, scaler_path=SCALER_PATH):
    scaler = load_scaler(scaler_path)
    x_encoded = preprocess_input(image_path, patient_age, patient_sex, view_position, scaler)
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
    MODEL_DIR = os.path.dirname(__file__)
    IMG_SIZE = (256, 256) 
    SCALER_PATH = os.path.join(MODEL_DIR, '../feature_scaler.pkl')

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
