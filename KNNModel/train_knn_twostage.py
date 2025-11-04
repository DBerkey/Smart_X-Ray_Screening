"""
Author: Douwe Berkeij
Date: 24-10-2025
Description: Two-stage KNN model for X-ray classification
    Stage 1: Binary classifier (Finding vs No Finding)
    Stage 2: Multi-label classifier (specific conditions)
AI use: in this code there was made use of GitHub Copilot to generate the docstrings and debugging
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from skimage.feature import hog
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, jaccard_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import concurrent.futures
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from PIL import Image
import gc
import pickle 

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
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=test_size + eval_size, random_state=42)
    x_test, x_eval, y_test, y_eval = train_test_split(x_temp, y_temp, test_size=eval_size / (test_size + eval_size), random_state=42)
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

def build_feature_matrix_batched(images_folder_path, df, img_size, batch_size=50):
    """
    Builds a feature matrix from images in batches to reduce memory usage.
    Uses a generator to yield batches instead of loading all images at once.
    
    Param:
        images_folder_path: Path to the folder containing images.
        df: DataFrame containing image metadata.
        img_size: Tuple specifying the desired image size (width, height).
        batch_size: Number of images to process in each batch.
    Yields:
        Batches of flattened images as numpy arrays.
    """
    num_images = len(df)
    image_indices = df["Image Index"].values
    
    print(f"Processing {num_images} images in batches of {batch_size}...")
    
    for i in range(0, num_images, batch_size):
        batch_end = min(i + batch_size, num_images)
        batch_images = []
        
        for img_index in image_indices[i:batch_end]:
            # Construct filename as soft_tissue_<image_index>
            img_name = f"soft_tissue_{img_index}"
            path = os.path.join(images_folder_path, img_name)
            batch_images.append(load_image_as_vector(path, img_size))
        
        print(f"  Processed batch {i//batch_size + 1}/{(num_images + batch_size - 1)//batch_size} ({batch_end}/{num_images} images)")
        yield np.array(batch_images, dtype=np.float32)
        
        # Free memory
        del batch_images
        gc.collect()

def extract_hog_features_batched(images_folder_path, df, img_size, batch_size=50, pixels_per_cell=(32, 32), cells_per_block=(4, 4), orientations=9):
    """
    Extracts HOG features from images in batches.
    Param:
        images_folder_path: Path to the folder containing images.
        df: DataFrame containing image metadata.
        img_size: Tuple specifying the desired image size (width, height).
        batch_size: Number of images to process in each batch.
        pixels_per_cell, cells_per_block, orientations: HOG parameters.
    Yields:
        Batches of HOG feature vectors as numpy arrays.
    """
    num_images = len(df)
    image_indices = df["Image Index"].values
    print(f"Extracting HOG features for {num_images} images in batches of {batch_size}...")

    for i in range(0, num_images, batch_size):
        batch_end = min(i + batch_size, num_images)
        batch_indices = image_indices[i:batch_end]
        args_list = [(img_index, images_folder_path, img_size, pixels_per_cell, cells_per_block, orientations) for img_index in batch_indices]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            batch_features = list(executor.map(process_image_hog, args_list))
        print(f"  Processed batch {i//batch_size + 1}/{(num_images + batch_size - 1)//batch_size} ({batch_end}/{num_images} images)")
        yield np.array(batch_features, dtype=np.float32)
        del batch_features
        gc.collect()

def get_hog_feature_matrix(images_folder_path, df, img_size, batch_size=50, pixels_per_cell=(32, 32), cells_per_block=(4, 4), orientations=9):
    """
    Builds the full HOG feature matrix from all images in batches.
    Returns:
        Feature matrix of shape (num_images, num_features)
    """
    print(f"\nBuilding HOG feature matrix for {len(df)} images...")
    feature_batches = []
    for batch in extract_hog_features_batched(images_folder_path, df, img_size, batch_size, pixels_per_cell, cells_per_block, orientations):
        feature_batches.append(batch)
    return np.vstack(feature_batches)

def process_image_hog(args):
    """
    Processes a single image to extract HOG features.
    Param:
        args: Tuple containing (img_index, images_folder_path, img_size, pixels_per_cell, cells_per_block, orientations)
    Returns:
        HOG feature vector as a numpy array.
    """
    img_index, images_folder_path, img_size, pixels_per_cell, cells_per_block, orientations = args
    img_name = f"soft_tissue_{img_index}"
    path = os.path.join(images_folder_path, img_name)
    img = Image.open(path).convert("L").resize(img_size)
    img_np = np.array(img) / 255.0
    features = hog(img_np, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, orientations=orientations, feature_vector=True)
    return features

def load_raw_feature_matrix(images_folder_path, df, img_size, batch_size=50):
    """
    Loads images in batches and returns the raw (flattened, scaled [0-1]) feature matrix.
    This is useful for fitting non-incremental models like LDA which require the full
    feature matrix in memory for the training split only.
    """
    print(f"\nLoading raw image matrix for {len(df)} images (batch_size={batch_size})...")
    batches = []
    total = 0
    for batch in build_feature_matrix_batched(images_folder_path, df, img_size, batch_size):
        batch = batch / 255.0
        batches.append(batch)
        total += len(batch)
        del batch
        gc.collect()
    X = np.vstack(batches) if len(batches) > 0 else np.empty((0, img_size[0]*img_size[1]), dtype=np.float32)
    print(f"Loaded raw feature matrix with shape: {X.shape}")
    return X

def create_binary_labels(y_series):
    """
    Creates binary labels for Stage 1: 1 if there are findings, 0 if "No Finding"
    Param:
        y_series: Series containing the finding labels.
    Returns:
        Binary labels (1 = has findings, 0 = no findings)
    """
    binary_labels = (y_series != 'No Finding').astype(int).values
    return binary_labels

def encode_predictor_labels(y_series, exclude_no_finding=True):
    """
    Encodes the labels using MultiLabelBinarizer.
    Param:
        y_series: Series containing the labels.
        exclude_no_finding: If True, excludes "No Finding" from the labels.
    Returns:
        Encoded labels and the MultiLabelBinarizer instance.
    """
    y_list = [labels.split('|') for labels in y_series]
    
    # Remove "No Finding" if requested
    if exclude_no_finding:
        y_list = [[label for label in labels if label != 'No Finding'] for labels in y_list]
    
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

    view_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    view_encoded = view_enc.fit_transform(df[['View Position']])
    X_combined = np.hstack([X_img, age_scaled, sex_encoded, view_encoded])

    return X_combined

def standardize_pca_features(X_train, X_test, X_eval=None):
    """
    Standardizes PCA-transformed features to have mean=0 and std=1.
    
    Param:
        X_train: Training features after PCA.
        X_test: Test features after PCA.
        X_eval: Evaluation features after PCA (optional).
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, X_eval_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_eval_scaled = scaler.transform(X_eval) if X_eval is not None else None
    return X_train_scaled, X_test_scaled, X_eval_scaled, scaler

def train_knn_model(X_train, Y_train, n_neighbors=5, metric='manhattan', weights='distance'):
    """
    Trains a KNN model.
    Param:
        X_train: Training features.
        Y_train: Training labels.
        n_neighbors: Number of neighbors to use.
        metric: Distance metric to use ('euclidean', 'manhattan', 'minkowski')
        weights: Weight function ('uniform' or 'distance')
    Returns:
        Trained KNN model.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
    knn.fit(X_train, Y_train)
    return knn

def train_svm_model(X_train, Y_train, config):
    """
    Trains an SVM model with the given config.
    Param:
        X_train: Training features.
        Y_train: Training labels.
        config: Dictionary of SVM parameters.
    Returns:
        Trained SVM model.
    """
    svm = SVC(**config)
    svm.fit(X_train, Y_train)
    return svm

def evaluate_binary_model(model, X_test, Y_test, name="Binary Model"):
    """
    Evaluates binary classification model.
    """
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        Y_test, Y_pred, average=None, zero_division=0
    )
    print(f"\n{name} Evaluation:")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    if isinstance(precision, (np.ndarray, list, tuple)) and len(precision) == 2:
        print(f"\n  Class 0 (No Finding):")
        print(f"    Precision: {precision[0]:.4f}")
        print(f"    Recall: {recall[0]:.4f}")
        print(f"    F1-Score: {f1[0]:.4f}")
        print(f"    Support: {support[0]}")
        print(f"\n  Class 1 (Has Finding):")
        print(f"    Precision: {precision[1]:.4f}")
        print(f"    Recall: {recall[1]:.4f}")
        print(f"    F1-Score: {f1[1]:.4f}")
        print(f"    Support: {support[1]}")
    else:
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Support: {support}")
    cm = confusion_matrix(Y_test, Y_pred)
    print(f"\n  Confusion Matrix:")
    print(cm)
    return accuracy, Y_pred, precision, recall, f1

def evaluate_multilabel_model(model, X_test, Y_test, name="Multi-label Model"):
    """
    Provides detailed evaluation metrics for multi-label classification.
    """    
    accuracy = model.score(X_test, Y_test)
    Y_pred = model.predict(X_test)
    
    hamming = hamming_loss(Y_test, Y_pred)
    f1_macro = f1_score(Y_test, Y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(Y_test, Y_pred, average='micro', zero_division=0)
    jaccard = jaccard_score(Y_test, Y_pred, average='samples', zero_division=0)
    
    print(f"\n{name} Evaluation:")
    print(f"  Exact Match Accuracy: {accuracy * 100:.2f}%")
    print(f"  Hamming Loss: {hamming:.4f}")
    print(f"  F1 Score (Macro): {f1_macro:.4f}")
    print(f"  F1 Score (Micro): {f1_micro:.4f}")
    print(f"  Jaccard Score: {jaccard:.4f}")
    
    return accuracy, hamming, f1_macro, f1_micro, jaccard, Y_pred

def evaluate_twostage_system(stage1_model, stage2_model, X_test, Y_test_binary, Y_test_multilabel, 
                              X_test_encoded, mlb_stage2, mlb_full, name="Two-Stage System"):
    """
    Evaluates the complete two-stage system.
    
    Param:
        stage1_model: Binary classifier (Finding vs No Finding)
        stage2_model: Multi-label classifier (specific conditions)
        X_test: Test features
        Y_test_binary: Binary test labels
        Y_test_multilabel: Multi-label test labels (one-hot encoded, includes "No Finding")
        X_test_encoded: Test features with additional metadata
        mlb_stage2: MultiLabelBinarizer for Stage 2 (excludes "No Finding")
        mlb_full: MultiLabelBinarizer for full labels (includes "No Finding")
    Returns:
        Overall metrics
    """
    print(f"\n{'='*60}")
    print(f"{name} - Complete Pipeline Evaluation")
    print(f"{'='*60}")
    
    # Stage 1: Predict if there are findings
    Y_stage1_pred = stage1_model.predict(X_test_encoded)
    
    # Count predictions
    no_finding_count = np.sum(Y_stage1_pred == 0)
    has_finding_count = np.sum(Y_stage1_pred == 1)
    
    print(f"\nStage 1 Predictions:")
    print(f"  No Finding: {no_finding_count} ({no_finding_count/len(Y_stage1_pred)*100:.1f}%)")
    print(f"  Has Finding: {has_finding_count} ({has_finding_count/len(Y_stage1_pred)*100:.1f}%)")
    
    # Initialize final predictions with zeros (all labels negative)
    Y_final_pred = np.zeros_like(Y_test_multilabel)
    
    # Stage 2: For cases with findings, predict specific conditions
    has_finding_indices = np.where(Y_stage1_pred == 1)[0]
    if len(has_finding_indices) > 0:
        X_has_findings = X_test_encoded[has_finding_indices]
        Y_stage2_pred = stage2_model.predict(X_has_findings)
        
        # Map Stage 2 predictions to full label space
        # Stage 2 mlb has labels excluding "No Finding"
        # Full mlb has all labels including "No Finding"
        for i, idx in enumerate(has_finding_indices):
            # Get the indices of the labels in the full mlb
            for j, label in enumerate(mlb_stage2.classes_):
                if label in mlb_full.classes_:
                    full_label_idx = np.where(mlb_full.classes_ == label)[0][0]
                    Y_final_pred[idx, full_label_idx] = Y_stage2_pred[i, j]
    
    # Calculate overall metrics
    accuracy = accuracy_score(
        np.argmax(Y_test_multilabel, axis=1) if Y_test_multilabel.shape[1] > 0 else Y_test_multilabel,
        np.argmax(Y_final_pred, axis=1) if Y_final_pred.shape[1] > 0 else Y_final_pred
    ) if Y_test_multilabel.shape[1] == 1 else np.mean(np.all(Y_test_multilabel == Y_final_pred, axis=1))
    
    hamming = hamming_loss(Y_test_multilabel, Y_final_pred)
    f1_macro = f1_score(Y_test_multilabel, Y_final_pred, average='macro', zero_division=0)
    f1_micro = f1_score(Y_test_multilabel, Y_final_pred, average='micro', zero_division=0)
    jaccard = jaccard_score(Y_test_multilabel, Y_final_pred, average='samples', zero_division=0)
    
    print(f"\nOverall Two-Stage System Performance:")
    print(f"  Exact Match Accuracy: {accuracy * 100:.2f}%")
    print(f"  Hamming Loss: {hamming:.4f}")
    print(f"  F1 Score (Macro): {f1_macro:.4f}")
    print(f"  F1 Score (Micro): {f1_micro:.4f}")
    print(f"  Jaccard Score: {jaccard:.4f}")
    
    # Stage 1 accuracy
    stage1_accuracy = accuracy_score(Y_test_binary, Y_stage1_pred)
    print(f"\nStage 1 (Binary) Accuracy: {stage1_accuracy * 100:.2f}%")
    
    return accuracy, hamming, f1_macro, f1_micro, jaccard, Y_final_pred

def fit_lda_model(X_train, Y_train, n_components=10):
    """
    Fits an LDA model for dimensionality reduction.
    Param:
        X_train: Training features.
        Y_train: Training labels.
        n_components: Number of components to keep.
    Returns:
        Fitted LDA model.
    """
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X_train, Y_train)
    return lda

if __name__ == "__main__":
    # ===== CONFIGURATION =====
    DATA_DIRECTORY_PATH = "C:/Users/berke/OneDrive/Documenten/school/UiA/Smart_X-Ray_Screening_img-data"
    images_folder_path = DATA_DIRECTORY_PATH + "/images_M-preprocessed"
    images_metadata_path = DATA_DIRECTORY_PATH + "/Data_Entry_2017_v2020.csv"
    
    # Image processing settings
    img_size = (128, 128)      # resolution of the images (width, height)
    n_pca_components = 1000     
    batch_size = 1000           # Must be >= n_pca_components for IncrementalPCA 
    
    LOAD_EXISTING_PCA = False    # Set to True to load existing PCA or LDA model
    LOAD_EXISTING_DATA = False   # Set to True to skip image loading and PCA or LDA transformation

    batch_size = 100
    processed_data_path = os.path.join(DATA_DIRECTORY_PATH, "processed_features.npz")
    scaler_path = os.path.join(DATA_DIRECTORY_PATH, "feature_scaler.pkl")

    # Read metadata and split data
    df = pd.read_csv(images_metadata_path)
    img_files = set(os.listdir(images_folder_path))
    df['ImageFile'] = df['Image Index'].apply(lambda idx: f"soft_tissue_{idx}")
    df = df[df['ImageFile'].isin(img_files)].reset_index(drop=True)
    x = df[['Image Index', 'Patient Age', 'Patient Sex', 'View Position']]
    y = df['Finding Labels']
    train, test, eval = split_data_train_test_eval(x, y)

    if LOAD_EXISTING_DATA and os.path.exists(processed_data_path) and os.path.exists(scaler_path):
        print("\n=== Loading Pre-processed Data ===")
        print(f"Loading from {processed_data_path}")
        loaded_data = np.load(processed_data_path)
        X_train_encoded = loaded_data['X_train_encoded']
        X_test_encoded = loaded_data['X_test_encoded']
        with open(scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)
        print(f"✓ Loaded preprocessed data: X_train shape = {X_train_encoded.shape}, X_test shape = {X_test_encoded.shape}")
    else:
        print("\n=== Extracting HOG Features for Training Data ===")
        X_train_hog = get_hog_feature_matrix(images_folder_path, pd.DataFrame(train[0]), img_size, batch_size=batch_size)
        print("\n=== Extracting HOG Features for Test Data ===")
        X_test_hog = get_hog_feature_matrix(images_folder_path, pd.DataFrame(test[0]), img_size, batch_size=batch_size)
        print("\n=== Standardizing HOG Features ===")
        X_train_std, X_test_std, _, feature_scaler = standardize_pca_features(X_train_hog, X_test_hog)
        print(f"Features standardized (mean≈0, std≈1)")
        X_train_encoded = encode_scale_input_features(pd.DataFrame(train[0]), X_train_std)
        X_test_encoded = encode_scale_input_features(pd.DataFrame(test[0]), X_test_std)
        print(f"\n=== Saving Processed Data ===")
        np.savez_compressed(processed_data_path, 
                            X_train_encoded=X_train_encoded,
                            X_test_encoded=X_test_encoded)
        print(f"✓ Saved to {processed_data_path}")
        print(f"✓ Saving feature scaler to {scaler_path}")
        with open(scaler_path, 'wb') as f:
            pickle.dump(feature_scaler, f)
            X_train_std, X_test_std, _, feature_scaler = standardize_pca_features(X_train_hog, X_test_hog)
            print(f"Features standardized (mean≈0, std≈1)")
            X_train_encoded = encode_scale_input_features(pd.DataFrame(train[0]), X_train_std)
            X_test_encoded = encode_scale_input_features(pd.DataFrame(test[0]), X_test_std)
            print(f"\n=== Saving Processed Data ===")
            np.savez_compressed(processed_data_path, 
                                X_train_encoded=X_train_encoded,
                                X_test_encoded=X_test_encoded)
            print(f"✓ Saved to {processed_data_path}")
            print(f"✓ Saving feature scaler to {scaler_path}")
            with open(scaler_path, 'wb') as f:
                pickle.dump(feature_scaler, f)
    print("\n" + "="*60)
    print("STAGE 1: BINARY CLASSIFICATION (Finding vs No Finding)")
    print("="*60)

    train_debug = (train[0], train[1])
    test_debug = (test[0], test[1])
    Y_train_binary = create_binary_labels(train_debug[1])
    Y_test_binary = create_binary_labels(test_debug[1])

    print(f"Training SVM on {len(X_train_encoded)} samples...")
    svm_config = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'probability': False}
    stage1_model = train_svm_model(X_train_encoded, Y_train_binary, svm_config)
    print("SVM training complete.")
    evaluate_binary_model(stage1_model, X_test_encoded, Y_test_binary, "Stage 1 SVM Model")
    # Save Stage 1 model
    stage1_model_path = os.path.join(DATA_DIRECTORY_PATH, "svm_stage1_binary.pkl")
    with open(stage1_model_path, 'wb') as f:
        pickle.dump(stage1_model, f)
    print(f"\n✓ Stage 1 model saved to {stage1_model_path}")

    # ===== STAGE 2: MULTI-LABEL CLASSIFICATION =====
    print("\n" + "="*60)
    print("STAGE 2: MULTI-LABEL CLASSIFICATION (Specific Conditions)")
    print("="*60)
    
    # Filter training data to only include cases with findings
    has_finding_train = train[1] != 'No Finding'
    X_train_stage2 = X_train_encoded[has_finding_train]
    y_train_stage2 = train[1][has_finding_train]
    
    has_finding_test = test[1] != 'No Finding'
    X_test_stage2 = X_test_encoded[has_finding_test]
    y_test_stage2 = test[1][has_finding_test]
    
    print(f"\nStage 2 Training samples: {len(X_train_stage2)}")
    print(f"Stage 2 Test samples: {len(X_test_stage2)}")
    
    # Encode labels (excluding "No Finding")
    Y_train_stage2, mlb = encode_predictor_labels(y_train_stage2, exclude_no_finding=True)
    Y_test_stage2, _ = encode_predictor_labels(y_test_stage2, exclude_no_finding=True)
    
    print(f"Number of condition labels: {len(mlb.classes_)}")
    print(f"Labels: {mlb.classes_}")
    
    # Test different configurations for Stage 2
    stage2_configs = [
        {'n_neighbors': 200, 'metric': 'euclidean', 'weights': 'distance'},
        {'n_neighbors': 500, 'metric': 'euclidean', 'weights': 'distance'},
        {'n_neighbors': 1000, 'metric': 'euclidean', 'weights': 'distance'},
        {'n_neighbors': 200, 'metric': 'manhattan', 'weights': 'distance'},
        {'n_neighbors': 500, 'metric': 'manhattan', 'weights': 'distance'},
        {'n_neighbors': 1000, 'metric': 'manhattan', 'weights': 'distance'},
    ]
    
    print(f"\nTesting {len(stage2_configs)} configurations for Stage 2...")
    
    best_stage2_f1_macro = 0
    best_stage2_config = None
    
    for i, config in enumerate(stage2_configs):
        print(f"\n[{i+1}/{len(stage2_configs)}] Testing: k={config['n_neighbors']}, metric={config['metric']}, weights={config['weights']}")
        
        knn_stage2 = train_knn_model(
            X_train_stage2, 
            Y_train_stage2,
            n_neighbors=config['n_neighbors'],
            metric=config['metric'],
            weights=config['weights']
        )
        
        Y_pred = knn_stage2.predict(X_test_stage2)
        f1_macro = f1_score(Y_test_stage2, Y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(Y_test_stage2, Y_pred, average='micro', zero_division=0)
        
        print(f"  F1 Macro: {f1_macro:.4f}, F1 Micro: {f1_micro:.4f}")
        
        if f1_macro > best_stage2_f1_macro:
            best_stage2_f1_macro = f1_macro
            best_stage2_config = config
    
    print("\n" + "="*60)
    print("BEST STAGE 2 CONFIGURATION")
    print("="*60)
    print(f"k={best_stage2_config['n_neighbors']}, metric={best_stage2_config['metric']}, weights={best_stage2_config['weights']}")
    print(f"F1 Macro: {best_stage2_f1_macro:.4f}")
    
    # Train final Stage 2 model
    print("\n=== Training Final Stage 2 Model ===")
    stage2_model = train_knn_model(
        X_train_stage2, Y_train_stage2,
        n_neighbors=best_stage2_config['n_neighbors'],
        metric=best_stage2_config['metric'],
        weights=best_stage2_config['weights']
    )
    
    evaluate_multilabel_model(stage2_model, X_test_stage2, Y_test_stage2, "Stage 2 Final Model")
    
    # Save Stage 2 model
    stage2_model_path = os.path.join(DATA_DIRECTORY_PATH, "knn_stage2_multilabel.pkl")
    with open(stage2_model_path, 'wb') as f:
        pickle.dump(stage2_model, f)
    print(f"\n✓ Stage 2 model saved to {stage2_model_path}")
    
    # Save label encoder
    mlb_path = os.path.join(DATA_DIRECTORY_PATH, "label_encoder_stage2.pkl")
    with open(mlb_path, 'wb') as f:
        pickle.dump(mlb, f)
    print(f"✓ Label encoder saved to {mlb_path}")
    
    # ===== EVALUATE COMPLETE TWO-STAGE SYSTEM =====
    print("\n" + "="*60)
    print("COMPLETE TWO-STAGE SYSTEM EVALUATION")
    print("="*60)
    
    # Create full multi-label encoding for complete evaluation
    Y_train_full, mlb_full = encode_predictor_labels(train[1], exclude_no_finding=False)
    Y_test_full, _ = encode_predictor_labels(test[1], exclude_no_finding=False)
    
    accuracy, hamming, f1_macro, f1_micro, jaccard, Y_final = evaluate_twostage_system(
        stage1_model, stage2_model, 
        X_test_encoded, Y_test_binary, Y_test_full,
        X_test_encoded, mlb, mlb_full,
        "Two-Stage KNN System"
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(X_train_encoded) + len(X_test_encoded)}")
    print(f"Training samples: {len(X_train_encoded)}")
    print(f"Test samples: {len(X_test_encoded)}")
    print(f"Image resolution: {img_size}")
    # PCA/LDA summary removed (not used)
    # Stage 1 config summary removed (not used)
    if best_stage2_config is not None:
        print(f"Stage 2 (Multi-label): k={best_stage2_config['n_neighbors']}, {best_stage2_config['metric']}")
    else:
        print("Stage 2 (Multi-label): No valid config found (likely due to small debug subset)")
    print(f"\nTwo-Stage System Performance:")
    print(f"  F1 Macro: {f1_macro:.4f}")
    print(f"  F1 Micro: {f1_micro:.4f}")
    print(f"  Hamming Loss: {hamming:.4f}")
    print("="*60)
