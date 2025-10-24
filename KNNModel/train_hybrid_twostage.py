"""
Author: Douwe Berkeij
Date: 24-10-2025
Description: Hybrid Two-stage model for X-ray classification
    Stage 1: Random Forest (Finding vs No Finding)
    Stage 2: KNN Multi-label classifier (specific conditions)
AI use: in this code there was made use of GitHub Copilot to generate the docstrings and debugging
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, jaccard_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
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
    
    X_combined = np.hstack([X_img, age_scaled, sex_encoded])
    return X_combined

def standardize_pca_features(X_train, X_test, X_eval=None):
    """
    Standardizes PCA-transformed features to have mean=0 and std=1.
    This is important for KNN to work properly.
    
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

def train_random_forest(X_train, Y_train, n_estimators=100, max_depth=None, class_weight='balanced'):
    """
    Trains a Random Forest model for binary classification.
    
    Param:
        X_train: Training features.
        Y_train: Training labels (binary).
        n_estimators: Number of trees.
        max_depth: Maximum depth of trees.
        class_weight: 'balanced' to handle class imbalance.
    Returns:
        Trained Random Forest model.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, Y_train)
    return rf

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
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights, n_jobs=-1)
    knn.fit(X_train, Y_train)
    return knn

def evaluate_binary_model(model, X_test, Y_test, name="Binary Model"):
    """
    Evaluates binary classification model.
    """
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    
    # Get precision, recall, f1 for both classes
    precision, recall, f1, support = precision_recall_fscore_support(
        Y_test, Y_pred, average=None, zero_division=0
    )
    
    print(f"\n{name} Evaluation:")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
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
    
    # Confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    [[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f"     [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    # Calculate balanced accuracy and F1
    balanced_acc = (recall[0] + recall[1]) / 2
    f1_avg = (f1[0] + f1[1]) / 2
    
    print(f"\n  Balanced Accuracy: {balanced_acc * 100:.2f}%")
    print(f"  Average F1: {f1_avg:.4f}")
    
    return accuracy, Y_pred, precision, recall, f1, balanced_acc

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

if __name__ == "__main__":
    # ===== CONFIGURATION =====
    DATA_DIRECTORY_PATH = "C:/Users/berke/OneDrive/Documenten/school/UiA/Smart_X-Ray_Screening_img-data"
    images_folder_path = DATA_DIRECTORY_PATH + "/images_M-preprocessed"
    images_metadata_path = DATA_DIRECTORY_PATH + "/Data_Entry_2017_v2020.csv"
    
    # Image processing settings
    img_size = (500, 500)      # resolution of the images (width, height)
    n_pca_components = 1000     
    batch_size = 1000           # Must be >= n_pca_components for IncrementalPCA 
    
    LOAD_EXISTING_PCA = True    # Set to True to load existing PCA model
    LOAD_EXISTING_DATA = True   # Set to True to skip image loading and PCA transformation
    
    # Stage 1 model choice: 'logistic' or 'random_forest'
    STAGE1_MODEL_TYPE = 'logistic'  # Change to 'random_forest' to try Random Forest
    # =========================
    
    print("\n" + "="*60)
    print("HYBRID TWO-STAGE CLASSIFICATION SYSTEM")
    print("="*60)
    print(f"Stage 1: {STAGE1_MODEL_TYPE.upper()} (Finding vs No Finding)")
    print("Stage 2: KNN Multi-label classification (Specific conditions)")
    print("="*60 + "\n")
    
    df = pd.read_csv(images_metadata_path)

    # Filter the metadata to only include images present in the folder
    image_files = set(os.listdir(images_folder_path))
    available_indices = set()
    for filename in image_files:
        if filename.startswith("soft_tissue_") and filename.endswith(".png"):
            img_index = filename[len("soft_tissue_"):]
            available_indices.add(img_index)
    
    df = df[df['Image Index'].astype(str).isin(available_indices)]
    
    # Print label distribution
    print("\n=== Label Distribution ===")
    no_finding_count = (df['Finding Labels'] == 'No Finding').sum()
    has_finding_count = (df['Finding Labels'] != 'No Finding').sum()
    print(f"No Finding: {no_finding_count} ({no_finding_count/len(df)*100:.1f}%)")
    print(f"Has Finding: {has_finding_count} ({has_finding_count/len(df)*100:.1f}%)")

    x = df[['Image Index', 'Patient Age', 'Patient Sex']]
    y = df['Finding Labels']

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
    scaler_path = os.path.join(DATA_DIRECTORY_PATH, "feature_scaler.pkl")
    
    if LOAD_EXISTING_DATA and os.path.exists(processed_data_path) and os.path.exists(pca_model_path) and os.path.exists(scaler_path):
        print("\n=== Loading Pre-processed Data ===")
        print(f"Loading from {processed_data_path}")
        
        loaded_data = np.load(processed_data_path)
        X_train_encoded = loaded_data['X_train_encoded']
        X_test_encoded = loaded_data['X_test_encoded']
        
        with open(pca_model_path, 'rb') as f:
            pca = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)
            
        print(f"✓ Loaded preprocessed data: X_train shape = {X_train_encoded.shape}, X_test shape = {X_test_encoded.shape}")
        print(f"✓ PCA explained variance: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")
        
    else:
        # Process from scratch
        if LOAD_EXISTING_PCA and os.path.exists(pca_model_path):
            print("\n=== Loading Existing PCA Model ===")
            with open(pca_model_path, 'rb') as f:
                pca = pickle.load(f)
            print(f"✓ Loaded PCA model with {pca.n_components} components")
            print(f"✓ Explained variance: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")
        else:
            print("\n=== Training PCA Model ===")
            # Fit PCA on training data using incremental approach
            pca = fit_pca_incremental(images_folder_path, pd.DataFrame(train[0]), img_size, 
                                      n_components=n_pca_components, batch_size=batch_size)
            # Save PCA model
            print(f"✓ Saving PCA model to {pca_model_path}")
            with open(pca_model_path, 'wb') as f:
                pickle.dump(pca, f)
        
        # Transform training data
        print("\n=== Processing Training Data ===")
        X_train = transform_features_batched(images_folder_path, pd.DataFrame(train[0]), 
                                             img_size, pca, batch_size=batch_size)
        
        print("\n=== Processing Test Data ===")
        X_test = transform_features_batched(images_folder_path, pd.DataFrame(test[0]), 
                                            img_size, pca, batch_size=batch_size)

        # Standardize PCA features
        print("\n=== Standardizing Features ===")
        X_train_std, X_test_std, _, feature_scaler = standardize_pca_features(X_train, X_test)
        print(f"Features standardized (mean≈0, std≈1)")

        # Combine with metadata
        X_train_encoded = encode_scale_input_features(pd.DataFrame(train[0]), X_train_std)
        X_test_encoded = encode_scale_input_features(pd.DataFrame(test[0]), X_test_std)
        
        # Save processed data
        print(f"\n=== Saving Processed Data ===")
        np.savez_compressed(processed_data_path, 
                           X_train_encoded=X_train_encoded,
                           X_test_encoded=X_test_encoded)
        print(f"✓ Saved to {processed_data_path}")
        
        # Save scaler
        print(f"✓ Saving feature scaler to {scaler_path}")
        with open(scaler_path, 'wb') as f:
            pickle.dump(feature_scaler, f)

    # ===== STAGE 1: BINARY CLASSIFICATION =====
    print("\n" + "="*60)
    print(f"STAGE 1: {STAGE1_MODEL_TYPE.upper()} BINARY CLASSIFICATION")
    print("="*60)
    
    # Create binary labels
    Y_train_binary = create_binary_labels(train[1])
    Y_test_binary = create_binary_labels(test[1])
    
    print(f"\nTraining set - No Finding: {np.sum(Y_train_binary == 0)}, Has Finding: {np.sum(Y_train_binary == 1)}")
    print(f"Test set - No Finding: {np.sum(Y_test_binary == 0)}, Has Finding: {np.sum(Y_test_binary == 1)}")
    
    # Test different Random Forest configurations
    print("\nTesting Random Forest configurations...")
    stage1_configs = [
        {'n_estimators': 50, 'max_depth': 10, 'class_weight': 'balanced'},
        {'n_estimators': 100, 'max_depth': 10, 'class_weight': 'balanced'},
        {'n_estimators': 100, 'max_depth': 20, 'class_weight': 'balanced'},
        {'n_estimators': 200, 'max_depth': 20, 'class_weight': 'balanced'},
        {'n_estimators': 100, 'max_depth': None, 'class_weight': 'balanced'},
    ]
    
    best_balanced_acc = 0
    best_stage1_config = None
    
    for i, config in enumerate(stage1_configs):
        print(f"\n[{i+1}/{len(stage1_configs)}] Testing: n_estimators={config['n_estimators']}, max_depth={config['max_depth']}")
        
        rf = train_random_forest(
            X_train_encoded, Y_train_binary,
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            class_weight=config['class_weight']
        )
        
        Y_pred = rf.predict(X_test_encoded)
        accuracy = accuracy_score(Y_test_binary, Y_pred)
        
        # Calculate balanced accuracy
        precision, recall, f1, support = precision_recall_fscore_support(
            Y_test_binary, Y_pred, average=None, zero_division=0
        )
        balanced_acc = (recall[0] + recall[1]) / 2
        
        print(f"  Accuracy: {accuracy * 100:.2f}%, Balanced Acc: {balanced_acc * 100:.2f}%")
        print(f"  Recall - No Finding: {recall[0]:.4f}, Has Finding: {recall[1]:.4f}")
        
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_stage1_config = config
    
    print("\n" + "="*60)
    print("BEST RANDOM FOREST CONFIGURATION")
    print("="*60)
    print(f"n_estimators={best_stage1_config['n_estimators']}, max_depth={best_stage1_config['max_depth']}")
    print(f"Balanced Accuracy: {best_balanced_acc * 100:.2f}%")
    
    # Train final model
    print("\n=== Training Final Stage 1 Model ===")
    stage1_model = train_random_forest(
        X_train_encoded, Y_train_binary,
        n_estimators=best_stage1_config['n_estimators'],
        max_depth=best_stage1_config['max_depth'],
        class_weight=best_stage1_config['class_weight']
    )
    
    # Evaluate Stage 1
    evaluate_binary_model(stage1_model, X_test_encoded, Y_test_binary, f"Stage 1 Final {STAGE1_MODEL_TYPE.upper()}")
    
    # Save Stage 1 model
    stage1_model_path = os.path.join(DATA_DIRECTORY_PATH, f"stage1_{STAGE1_MODEL_TYPE}.pkl")
    with open(stage1_model_path, 'wb') as f:
        pickle.dump(stage1_model, f)
    print(f"\n✓ Stage 1 model saved to {stage1_model_path}")
    
    # ===== STAGE 2: MULTI-LABEL CLASSIFICATION =====
    print("\n" + "="*60)
    print("STAGE 2: KNN MULTI-LABEL CLASSIFICATION")
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
    
    # Test different KNN configurations for Stage 2
    stage2_configs = [
        {'n_neighbors': 50, 'metric': 'euclidean', 'weights': 'distance'},
        {'n_neighbors': 100, 'metric': 'euclidean', 'weights': 'distance'},
        {'n_neighbors': 200, 'metric': 'euclidean', 'weights': 'distance'},
        {'n_neighbors': 100, 'metric': 'manhattan', 'weights': 'distance'},
        {'n_neighbors': 200, 'metric': 'manhattan', 'weights': 'distance'},
    ]
    
    print(f"\nTesting {len(stage2_configs)} KNN configurations for Stage 2...")
    
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
    print("BEST STAGE 2 KNN CONFIGURATION")
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
    
    evaluate_multilabel_model(stage2_model, X_test_stage2, Y_test_stage2, "Stage 2 Final KNN")
    
    # Save Stage 2 model
    stage2_model_path = os.path.join(DATA_DIRECTORY_PATH, "stage2_knn_multilabel.pkl")
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
        "Hybrid Two-Stage System"
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(X_train_encoded) + len(X_test_encoded)}")
    print(f"Training samples: {len(X_train_encoded)}")
    print(f"Test samples: {len(X_test_encoded)}")
    print(f"Image resolution: {img_size}")
    print(f"PCA components: {n_pca_components}")
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")
    print(f"\nStage 1: {STAGE1_MODEL_TYPE.upper()}")
    if STAGE1_MODEL_TYPE == 'logistic':
        print(f"  C={best_stage1_config['C']}, class_weight={best_stage1_config['class_weight']}")
    else:
        print(f"  n_estimators={best_stage1_config['n_estimators']}, max_depth={best_stage1_config['max_depth']}")
    print(f"\nStage 2: KNN")
    print(f"  k={best_stage2_config['n_neighbors']}, metric={best_stage2_config['metric']}")
    print(f"\nTwo-Stage System Performance:")
    print(f"  F1 Macro: {f1_macro:.4f}")
    print(f"  F1 Micro: {f1_micro:.4f}")
    print(f"  Hamming Loss: {hamming:.4f}")
    print("="*60)
