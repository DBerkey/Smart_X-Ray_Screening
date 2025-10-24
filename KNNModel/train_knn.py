"""
Author: Douwe Berkeij
Date: 13-10-2025
AI use: in this code there was made use of GitHub Copilot to generate the docstrings and debuging
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score, mean_squared_error, hamming_loss, f1_score, jaccard_score
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
        Flattened image as a numpy array.
    """
    X_features = []
    for img_index in df["Image Index"]:
        # Construct filename as soft_tissue_<image_index>
        img_name = f"soft_tissue_{img_index}"
        path = os.path.join(images_folder_path, img_name)
        X_features.append(load_image_as_vector(path, img_size))
    return np.array(X_features)

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

def standardize_pca_features(X_train, X_test):
    """
    Standardizes PCA-transformed features to have mean=0 and std=1.
    This is important for KNN to work properly.
    
    Param:
        X_train: Training features after PCA.
        X_test: Test features after PCA.
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

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
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
    knn.fit(X_train, Y_train)
    return knn

def evaluate_model(model, X_test, Y_test):
    """
    Evaluates the model on the test set.
    Param:
        model: Trained model.
        X_test: Test features.
        Y_test: Test labels.
    Returns:
        Accuracy of the model on the test set.
    """
    accuracy = model.score(X_test, Y_test)
    return accuracy

def evaluate_model_detailed(model, X_test, Y_test, name="Model"):
    """
    Provides detailed evaluation metrics for multi-label classification.
    """    
    accuracy = model.score(X_test, Y_test)
    Y_pred = model.predict(X_test)
    
    hamming = hamming_loss(Y_test, Y_pred)
    f1_macro = f1_score(Y_test, Y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(Y_test, Y_pred, average='micro', zero_division=0)
    jaccard = jaccard_score(Y_test, Y_pred, average='samples', zero_division=0)
    mse = mean_squared_error(Y_test, Y_pred)

    
    print(f"\n{name} Evaluation:")
    print(f"  Exact Match Accuracy: {accuracy * 100:.2f}%")
    print(f"  Hamming Loss: {hamming:.4f}")
    print(f"  F1 Score (Macro): {f1_macro:.4f}")
    print(f"  F1 Score (Micro): {f1_micro:.4f}")
    print(f"  Jaccard Score: {jaccard:.4f}")
    
    return accuracy, mse, hamming, f1_macro, f1_micro, jaccard

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
    # =========================
    
    df = pd.read_csv(images_metadata_path)

    # Filter the metadata to only include images present in the folder
    # Image files are named as soft_tissue_<image_index>.png
    image_files = set(os.listdir(images_folder_path))
    available_indices = set()
    for filename in image_files:
        if filename.startswith("soft_tissue_") and filename.endswith(".png"):
            img_index = filename[len("soft_tissue_"):]
            available_indices.add(img_index)
    
    df = df[df['Image Index'].astype(str).isin(available_indices)]

    x = df[['Image Index', 'Patient Age', 'Patient Sex']]
    y = df['Finding Labels']

    train, test, eval = split_data_train_test_eval(x, y)

    # Save evaluation data to CSV for application use
    eval_csv_path = os.path.join(DATA_DIRECTORY_PATH, "eval_data.csv")
    eval_df = pd.DataFrame(eval[0])
    eval_df['Finding Labels'] = eval[1].values
    eval_df.to_csv(eval_csv_path, index=False)
    print(f"Evaluation data saved to {eval_csv_path}")

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
        
        # Transform training data
        print("\n=== Processing Training Data ===")
        X_train = transform_features_batched(images_folder_path, pd.DataFrame(train[0]), 
                                             img_size, pca, batch_size=batch_size)
        
        print("\n=== Processing Test Data ===")
        # Transform test data using the same PCA
        X_test = transform_features_batched(images_folder_path, pd.DataFrame(test[0]), 
                                            img_size, pca, batch_size=batch_size)

        # Standardize PCA features (important for KNN!)
        print("\n=== Standardizing Features ===")
        X_train_std, X_test_std, feature_scaler = standardize_pca_features(X_train, X_test)
        print(f"Features standardized (mean≈0, std≈1)")

        Y_train, mlb = encode_predictor_labels(train[1])
        Y_test, _ = encode_predictor_labels(test[1])

        X_train_encoded = encode_scale_input_features(pd.DataFrame(train[0]), X_train_std)
        X_test_encoded = encode_scale_input_features(pd.DataFrame(test[0]), X_test_std)
        
        # Save processed data for future runs
        print(f"\n=== Saving Processed Data for Future Runs ===")
        np.savez_compressed(processed_data_path, 
                           X_train_encoded=X_train_encoded,
                           X_test_encoded=X_test_encoded)
        print(f"✓ Saved to {processed_data_path}")
    
    # Load labels (always needed)
    Y_train, mlb = encode_predictor_labels(train[1])
    Y_test, _ = encode_predictor_labels(test[1])

    # Save PCA model and scaler for later use (if they were just created)
    # Save PCA model if it was just trained (not loaded)
    if not (LOAD_EXISTING_PCA and os.path.exists(pca_model_path)):
        print(f"\n✓ Saving PCA model to {pca_model_path}")
        with open(pca_model_path, 'wb') as f:
            pickle.dump(pca, f)
    
    # Save scaler and processed data if they were just created (not loaded)
    if not (LOAD_EXISTING_DATA and os.path.exists(processed_data_path)):
        print(f"✓ Saving feature scaler to {scaler_path}")
        with open(scaler_path, 'wb') as f:
            pickle.dump(feature_scaler, f)


    print("\n=== Finding Best KNN Configuration ===")
    # Test different configurations on a subset to find the best
    test_configs = [
        {'n_neighbors': 200, 'metric': 'euclidean', 'weights': 'distance'},
        {'n_neighbors': 200, 'metric': 'manhattan', 'weights': 'distance'},
        {'n_neighbors': 200, 'metric': 'minkowski', 'weights': 'distance'},
        {'n_neighbors': 1000, 'metric': 'euclidean', 'weights': 'distance'},
        {'n_neighbors': 1000, 'metric': 'manhattan', 'weights': 'distance'},
        {'n_neighbors': 1000, 'metric': 'minkowski', 'weights': 'distance'},
        {'n_neighbors': 1500, 'metric': 'euclidean', 'weights': 'distance'},
        {'n_neighbors': 1500, 'metric': 'manhattan', 'weights': 'distance'},
        {'n_neighbors': 1500, 'metric': 'minkowski', 'weights': 'distance'},
        {'n_neighbors': 2000, 'metric': 'euclidean', 'weights': 'distance'},
        {'n_neighbors': 2000, 'metric': 'manhattan', 'weights': 'distance'},
        {'n_neighbors': 2000, 'metric': 'minkowski', 'weights': 'distance'}
    ]   
    
    print(f"Testing {len(test_configs)} configurations...")
    
    best_accuracy = 0
    best_config = None
    
    for i, config in enumerate(test_configs):
        print(f"\n[{i+1}/{len(test_configs)}] Testing: k={config['n_neighbors']}, metric={config['metric']}, weights={config['weights']}")
        
        knn_test = train_knn_model(
            X_train_encoded, 
            Y_train,
            n_neighbors=config['n_neighbors'],
            metric=config['metric'],
            weights=config['weights']
        )
        
        test_accuracy = evaluate_model(knn_test, X_test_encoded, Y_test)
        print(f"  Accuracy: {test_accuracy * 100:.2f}%")
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_config = config
    
    print("\n" + "="*60)
    print("BEST CONFIGURATION FOUND")
    print("="*60)
    print(f"k={best_config['n_neighbors']}, metric={best_config['metric']}, weights={best_config['weights']}")
    print(f"Subset accuracy: {best_accuracy * 100:.2f}%")
    
    print("\n=== Training Final KNN Model with Best Configuration ===")
    knn_model = train_knn_model(
        X_train_encoded, Y_train, 
        n_neighbors=best_config['n_neighbors'],
        metric=best_config['metric'],
        weights=best_config['weights']
    )
    
    # Detailed evaluation
    accuracy, mse, hamming, f1_macro, f1_micro, jaccard = evaluate_model_detailed(knn_model, X_test_encoded, Y_test, "Final KNN Model")
    print(f"\nConfiguration: k={best_config['n_neighbors']}, metric={best_config['metric']}, weights={best_config['weights']}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"Jaccard Score: {jaccard:.4f}")

    # Save KNN model
    knn_model_path = os.path.join(DATA_DIRECTORY_PATH, "knn_model.pkl")
    with open(knn_model_path, 'wb') as f:
        pickle.dump(knn_model, f)
    print(f"KNN model saved to {knn_model_path}")
    
    # Save label encoder
    mlb_path = os.path.join(DATA_DIRECTORY_PATH, "label_encoder.pkl")
    with open(mlb_path, 'wb') as f:
        pickle.dump(mlb, f)
    print(f"Label encoder saved to {mlb_path}")
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Images processed: {len(X_train_encoded) + len(X_test_encoded)}")
    print(f"Training samples: {len(X_train_encoded)}")
    print(f"Test samples: {len(X_test_encoded)}")
    print(f"Image resolution: {img_size}")
    print(f"PCA components: {n_pca_components}")
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")
    print(f"Final features per sample: {X_train_encoded.shape[1]}")
    print(f"Number of labels: {Y_train.shape[1]}")
    print(f"Best KNN configuration: k={best_config['n_neighbors']}, metric={best_config['metric']}, weights={best_config['weights']}")
    print(f"Final accuracy: {accuracy * 100:.2f}%")
    print("="*60)
    