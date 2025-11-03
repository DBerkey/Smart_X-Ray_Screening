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
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, jaccard_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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

def fit_pca_incremental(images_folder_path, df, img_size, n_components=500, batch_size=50):
    """
    Fits an IncrementalPCA model on images using batch processing.
    This avoids loading all images into memory at once.
    
    Param:
        images_folder_path: Path to the folder containing images.
        df: DataFrame containing image metadata.
        img_size: Tuple specifying the desired image size (width, height).
        n_components: Number of principal components to keep.
        batch_size: Number of images to process in each batch.
    Returns:
        Fitted IncrementalPCA model.
    """
    print(f"\nFitting Incremental PCA with {n_components} components on {len(df)} images...")
    
    # Initialize IncrementalPCA
    pca = IncrementalPCA(n_components=n_components)

    batch_count = 0
    total_samples = 0
    
    for batch in build_feature_matrix_batched(images_folder_path, df, img_size, batch_size):
        batch = batch / 255.0
        
        batch_count += 1
        total_samples += len(batch)
        
        pca.partial_fit(batch)
        
        if batch_count % 10 == 0 or batch_count == 1:
            print(f"  Fitted batch {batch_count} ({total_samples}/{len(df)} samples)")
        
        del batch
        gc.collect()
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA fitted on ALL {total_samples} samples")
    print(f"Explained variance ratio: {explained_var:.4f} ({explained_var*100:.2f}%)")
    
    if explained_var < 0.80:
        print(f"WARNING: Only {explained_var*100:.1f}% variance captured!")
    
    return pca

def transform_features_batched(images_folder_path, df, img_size, pca, batch_size=50):
    """
    Transforms images using a fitted PCA model, processing in batches.
    
    Param:
        images_folder_path: Path to the folder containing images.
        df: DataFrame containing image metadata.
        img_size: Tuple specifying the desired image size (width, height).
        pca: Fitted PCA model.
        batch_size: Number of images to process in each batch.
    Returns:
        Transformed feature matrix (reduced dimensions).
    """
    print(f"\nTransforming {len(df)} images using PCA...")
    
    transformed_batches = []
    batch_count = 0
    
    for batch in build_feature_matrix_batched(images_folder_path, df, img_size, batch_size):
        batch = batch / 255.0
        
        transformed_batch = pca.transform(batch)
        transformed_batches.append(transformed_batch)
        
        batch_count += 1
        if batch_count % 20 == 0:
            print(f"  Transformed {batch_count * batch_size}/{len(df)} images")
        
        del batch
        gc.collect()
    
    X_transformed = np.vstack(transformed_batches)
    print(f"Transformation complete. Final shape: {X_transformed.shape}")
    return X_transformed

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
    img_size = (500, 500)      # resolution of the images (width, height)
    n_pca_components = 1000     
    batch_size = 1000           # Must be >= n_pca_components for IncrementalPCA 
    
    LOAD_EXISTING_PCA = True    # Set to True to load existing PCA or LDA model
    LOAD_EXISTING_DATA = True   # Set to True to skip image loading and PCA or LDAtransformation

    PCAUSE = False              # Set to True to use PCA instead of LDA
    # =========================
    
    print("\n" + "="*60)
    print("TWO-STAGE KNN CLASSIFICATION SYSTEM")
    print("="*60)
    print("Stage 1: Binary classification (Finding vs No Finding)")
    print("Stage 2: Multi-label classification (Specific conditions)")
    print("="*60 + "\n")
    
    df = pd.read_csv(images_metadata_path)

    # Filter the metadata to only include images present as .png files
    img_dir = images_folder_path
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    # Remove 'soft_tissue_' prefix and '.png' from filenames to get indices
    available_indices = [f.replace('soft_tissue_', '').replace('.png', '') for f in img_files]
    # Remove .png from CSV and match full numeric part with suffix
    df['Image Index Num'] = df['Image Index'].str.replace('.png', '')  # 00000001_000.png -> 00000001_000
    df = df[df['Image Index Num'].isin(available_indices)].reset_index(drop=True)
    
    # Print label distribution
    print("\n=== Label Distribution ===")
    no_finding_count = (df['Finding Labels'] == 'No Finding').sum()
    has_finding_count = (df['Finding Labels'] != 'No Finding').sum()
    print(f"No Finding: {no_finding_count} ({no_finding_count/len(df)*100:.1f}%)")
    print(f"Has Finding: {has_finding_count} ({has_finding_count/len(df)*100:.1f}%)")

    x = df[['Image Index', 'Patient Age', 'Patient Sex', 'View Position']]
    y = df['Finding Labels']
    if len(x) == 0 or len(y) == 0:
        raise ValueError("No samples found after filtering metadata for available .npy files. Check your file naming and metadata consistency.")
    train, test, eval = split_data_train_test_eval(x, y)

    # Save evaluation data to CSV for application use
    eval_csv_path = os.path.join(DATA_DIRECTORY_PATH, "eval_data.csv")
    eval_df = pd.DataFrame(eval[0])
    eval_df['Finding Labels'] = eval[1].values
    eval_df.to_csv(eval_csv_path, index=False)
    print(f"\nEvaluation data saved to {eval_csv_path}")

    # Check if we should load existing processed data
    processed_data_path = os.path.join(DATA_DIRECTORY_PATH, "processed_features.npz")
    pca_model_path = os.path.join(DATA_DIRECTORY_PATH, "pca_model.pkl")
    lda_model_path = os.path.join(DATA_DIRECTORY_PATH, "lda_model.pkl")
    scaler_path = os.path.join(DATA_DIRECTORY_PATH, "feature_scaler.pkl")
    
    if LOAD_EXISTING_DATA and os.path.exists(processed_data_path) and os.path.exists(scaler_path):
        print("\n=== Loading Pre-processed Data ===")
        print(f"Loading from {processed_data_path}")
        
        loaded_data = np.load(processed_data_path)
        X_train_encoded = loaded_data['X_train_encoded']
        X_test_encoded = loaded_data['X_test_encoded']
        
        if PCAUSE:
            with open(pca_model_path, 'rb') as f:
                pca = pickle.load(f)
        else:
            with open(lda_model_path, 'rb') as f:
                lda = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)
            
        print(f"✓ Loaded preprocessed data: X_train shape = {X_train_encoded.shape}, X_test shape = {X_test_encoded.shape}")
        if PCAUSE:
            try:
                explained = pca.explained_variance_ratio_.sum()
                print(f"✓ PCA explained variance: {explained:.4f} ({explained*100:.2f}%)")
            except Exception:
                print("✓ PCA model loaded (could not read explained variance)")
        else:
            print(f"✓ LDA model loaded (n_components={getattr(lda, 'n_components', 'unknown')})")
        
    else:
        # Process from scratch
        if PCAUSE:
            if LOAD_EXISTING_PCA and os.path.exists(pca_model_path):
                print("\n=== Loading Existing PCA Model ===")
                with open(pca_model_path, 'rb') as f:
                    pca = pickle.load(f)
                print(f"✓ Loaded PCA model with {pca.n_components} components")
                try:
                    explained = pca.explained_variance_ratio_.sum()
                    print(f"✓ Explained variance: {explained:.4f} ({explained*100:.2f}%)")
                except Exception:
                    pass
            else:
                print("\n=== Training PCA Model ===")
                # Fit PCA on training data using incremental approach
                pca = fit_pca_incremental(images_folder_path, pd.DataFrame(train[0]), img_size, 
                                          n_components=n_pca_components, batch_size=batch_size)
                # Save PCA model
                print(f"✓ Saving PCA model to {pca_model_path}")
                with open(pca_model_path, 'wb') as f:
                    pickle.dump(pca, f)

            # Transform training and test data using PCA
            print("\n=== Processing Training Data (PCA) ===")
            X_train = transform_features_batched(images_folder_path, pd.DataFrame(train[0]), 
                                                 img_size, pca, batch_size=batch_size)
            
            print("\n=== Processing Test Data (PCA) ===")
            X_test = transform_features_batched(images_folder_path, pd.DataFrame(test[0]), 
                                                img_size, pca, batch_size=batch_size)

            # Standardize PCA features
            print("\n=== Standardizing Features ===")
            X_train_std, X_test_std, _, feature_scaler = standardize_pca_features(X_train, X_test)
            print(f"Features standardized (mean≈0, std≈1)")

            # Combine with metadata
            X_train_encoded = encode_scale_input_features(pd.DataFrame(train[0]), X_train_std)
            X_test_encoded = encode_scale_input_features(pd.DataFrame(test[0]), X_test_std)

            # Save processed data and scaler
            print(f"\n=== Saving Processed Data ===")
            np.savez_compressed(processed_data_path, 
                               X_train_encoded=X_train_encoded,
                               X_test_encoded=X_test_encoded)
            print(f"✓ Saved to {processed_data_path}")
            print(f"✓ Saving feature scaler to {scaler_path}")
            with open(scaler_path, 'wb') as f:
                pickle.dump(feature_scaler, f)

        else:
            # LDA path - First reduce dimensions with PCA, then apply LDA
            # This avoids memory issues with high-dimensional data
            print("\n=== Preparing data for PCA+LDA (two-step dimensionality reduction) ===")
            
            # Step 1: Apply PCA first to reduce dimensions
            if LOAD_EXISTING_PCA and os.path.exists(pca_model_path):
                print("\n=== Loading Existing PCA Model ===")
                with open(pca_model_path, 'rb') as f:
                    pca = pickle.load(f)
                print(f"✓ Loaded PCA model with {pca.n_components} components")
                try:
                    explained = pca.explained_variance_ratio_.sum()
                    print(f"✓ Explained variance: {explained:.4f} ({explained*100:.2f}%)")
                except Exception:
                    pass
            else:
                print("\n=== Training PCA Model (Step 1 of PCA+LDA) ===")
                # Fit PCA on training data using incremental approach
                pca = fit_pca_incremental(images_folder_path, pd.DataFrame(train[0]), img_size, 
                                          n_components=n_pca_components, batch_size=batch_size)
                # Save PCA model
                print(f"✓ Saving PCA model to {pca_model_path}")
                with open(pca_model_path, 'wb') as f:
                    pickle.dump(pca, f)

            # Transform data using PCA
            print("\n=== Processing Training Data with PCA ===")
            X_train_pca = transform_features_batched(images_folder_path, pd.DataFrame(train[0]), 
                                                 img_size, pca, batch_size=batch_size)
            
            print("\n=== Processing Test Data with PCA ===")
            X_test_pca = transform_features_batched(images_folder_path, pd.DataFrame(test[0]), 
                                                img_size, pca, batch_size=batch_size)

            # Step 2: Apply LDA on PCA-reduced features
            print("\n=== Applying LDA on PCA-reduced features (Step 2 of PCA+LDA) ===")
            Y_train_binary = create_binary_labels(train[1])

            # Determine allowed number of components for LDA: at most n_classes-1
            unique_classes = np.unique(Y_train_binary)
            max_lda_components = max(1, len(unique_classes) - 1)
            # LDA components cannot exceed n_classes-1 (which is 1 for binary classification)
            n_lda_components = min(max_lda_components, 1)

            print(f"Fitting LDA with n_components={n_lda_components} (max allowed: {max_lda_components}) on PCA-reduced data with shape {X_train_pca.shape}")
            lda = LinearDiscriminantAnalysis(n_components=n_lda_components)
            lda.fit(X_train_pca, Y_train_binary)

            # Transform data using LDA
            X_train_lda = lda.transform(X_train_pca)
            X_test_lda = lda.transform(X_test_pca)

            # Standardize LDA features
            print("\n=== Standardizing LDA Features ===")
            X_train_std, X_test_std, _, feature_scaler = standardize_pca_features(X_train_lda, X_test_lda)

            # Combine with metadata
            X_train_encoded = encode_scale_input_features(pd.DataFrame(train[0]), X_train_std)
            X_test_encoded = encode_scale_input_features(pd.DataFrame(test[0]), X_test_std)

            # Save processed data and LDA model
            print(f"\n=== Saving Processed Data and LDA model ===")
            np.savez_compressed(processed_data_path, 
                               X_train_encoded=X_train_encoded,
                               X_test_encoded=X_test_encoded)
            print(f"✓ Saved to {processed_data_path}")
            print(f"✓ Saving feature scaler to {scaler_path}")
            with open(scaler_path, 'wb') as f:
                pickle.dump(feature_scaler, f)

            print(f"✓ Saving LDA model to {lda_model_path}")
            with open(lda_model_path, 'wb') as f:
                pickle.dump(lda, f)

    print("\n=== t-SNE Visualization (Binary Labels) ===")
    # Use the encoded features after dimensionality reduction (X_train_encoded)
    # For speed, use a subset if dataset is large
    tsne_sample_size = min(2000, len(X_train_encoded))
    idx = np.random.choice(len(X_train_encoded), tsne_sample_size, replace=False)
    X_vis = X_train_encoded[idx]
    y_vis = Y_train_binary[idx]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_vis)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_vis, cmap='coolwarm', alpha=0.6)
    plt.title('t-SNE Visualization of Training Data (Binary Labels)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.tight_layout()
    plt.show()

    # ===== STAGE 1: BINARY CLASSIFICATION =====
    print("\n" + "="*60)
    print("STAGE 1: BINARY CLASSIFICATION (Finding vs No Finding)")
    print("="*60)
    
    # Create binary labels
    Y_train_binary = create_binary_labels(train[1])
    Y_test_binary = create_binary_labels(test[1])
    
    print(f"\nTraining set - No Finding: {np.sum(Y_train_binary == 0)}, Has Finding: {np.sum(Y_train_binary == 1)}")
    print(f"Test set - No Finding: {np.sum(Y_test_binary == 0)}, Has Finding: {np.sum(Y_test_binary == 1)}")
    

    # ===== SVM AS MAIN CLASSIFIER FOR STAGE 1 =====
    print("\n" + "="*60)
    print("STAGE 1: SVM CLASSIFIER")
    print("="*60)
    svm_config = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'probability': False}
    print(f"Training SVM with config: {svm_config}")
    stage1_model = train_svm_model(X_train_encoded, Y_train_binary, svm_config)
    Y_pred_svm = stage1_model.predict(X_test_encoded)
    accuracy_svm = accuracy_score(Y_test_binary, Y_pred_svm)
    f1_svm = f1_score(Y_test_binary, Y_pred_svm, average='weighted')
    print(f"SVM Accuracy: {accuracy_svm * 100:.2f}%, F1 (weighted): {f1_svm:.4f}")
    evaluate_binary_model(stage1_model, X_test_encoded, Y_test_binary, "Stage 1 SVM Model")
    # Save Stage 1 SVM model
    stage1_model_path = os.path.join(DATA_DIRECTORY_PATH, "svm_stage1_binary.pkl")
    with open(stage1_model_path, 'wb') as f:
        pickle.dump(stage1_model, f)
    print(f"\n✓ Stage 1 SVM model saved to {stage1_model_path}")
    
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
    if PCAUSE:
        print(f"PCA components: {n_pca_components}")
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")
    else:
        print(f"Dimensionality reduction: PCA (n={n_pca_components}) + LDA")
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")
    print(f"\nStage 1 (Binary): k={best_stage1_config['n_neighbors']}, {best_stage1_config['metric']}")
    print(f"Stage 2 (Multi-label): k={best_stage2_config['n_neighbors']}, {best_stage2_config['metric']}")
    print(f"\nTwo-Stage System Performance:")
    print(f"  F1 Macro: {f1_macro:.4f}")
    print(f"  F1 Micro: {f1_micro:.4f}")
    print(f"  Hamming Loss: {hamming:.4f}")
    print("="*60)
