"""
Author: Douwe Berkeij
Date: 13-10-2025
AI use: in this code there was made use of GitHub Copilot to generate the docstrings and debuging
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.decomposition import IncrementalPCA
import pandas as pd
import os
import numpy as np
from PIL import Image
import gc  # For garbage collection to free memory
import pickle  # For saving models

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
    for img_name in df["Image Index"]:
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
    image_names = df["Image Index"].values
    
    print(f"Processing {num_images} images in batches of {batch_size}...")
    
    for i in range(0, num_images, batch_size):
        batch_end = min(i + batch_size, num_images)
        batch_images = []
        
        for img_name in image_names[i:batch_end]:
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
    print(f"\nFitting Incremental PCA with {n_components} components...")
    
    # Initialize IncrementalPCA (don't set batch_size parameter)
    pca = IncrementalPCA(n_components=n_components)
    
    # For the first batch, we need at least n_components samples
    # Accumulate batches until we have enough
    first_batch_data = []
    min_samples_needed = n_components
    samples_collected = 0
    
    batch_generator = build_feature_matrix_batched(images_folder_path, df, img_size, batch_size)
    
    # Collect first batches to meet minimum sample requirement
    print(f"Collecting initial samples (need at least {min_samples_needed})...")
    for batch in batch_generator:
        first_batch_data.append(batch)
        samples_collected += len(batch)
        
        if samples_collected >= min_samples_needed:
            # Combine accumulated batches and do first fit
            combined_batch = np.vstack(first_batch_data)
            print(f"Fitting PCA on initial {len(combined_batch)} samples...")
            pca.partial_fit(combined_batch)
            
            # Free memory
            del first_batch_data
            del combined_batch
            gc.collect()
            break
    
    # Continue with remaining batches
    batch_count = 1
    for batch in batch_generator:
        batch_count += 1
        pca.partial_fit(batch)
        if batch_count % 10 == 0:
            print(f"  Fitted {batch_count} additional batches...")
        del batch
        gc.collect()
    
    print(f"PCA fitted on all data. Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
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
    print(f"\nTransforming images using PCA...")
    
    transformed_batches = []
    for batch in build_feature_matrix_batched(images_folder_path, df, img_size, batch_size):
        transformed_batch = pca.transform(batch)
        transformed_batches.append(transformed_batch)
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

def train_knn_model(X_train, Y_train, n_neighbors=5):
    """
    Trains a KNN model.
    Param:
        X_train: Training features.
        Y_train: Training labels.
        n_neighbors: Number of neighbors to use.
    Returns:
        Trained KNN model.
    """
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
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

if __name__ == "__main__":
    # ===== CONFIGURATION =====
    DATA_DIRECTORY_PATH = "C:/Users/berke/OneDrive/Documenten/school/UiA/Smart_X-Ray_Screening_img-data"
    images_folder_path = DATA_DIRECTORY_PATH + "/images_M"
    images_metadata_path = DATA_DIRECTORY_PATH + "/Data_Entry_2017_v2020.csv"
    
    # Image processing settings
    img_size = (1000, 820)      # resolution of the images (width, height)
    n_pca_components = 500      # Reduce dimensions to this number
    batch_size = 550            # Must be >= n_pca_components for IncrementalPCA 
    # =========================
    
    # Validate configuration
    if batch_size < n_pca_components:
        batch_size = n_pca_components + 50

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

    print("\n=== Processing Training Data ===")
    # Fit PCA on training data using incremental approach
    pca = fit_pca_incremental(images_folder_path, pd.DataFrame(train[0]), img_size, 
                              n_components=n_pca_components, batch_size=batch_size)
    
    # Transform training data
    X_train = transform_features_batched(images_folder_path, pd.DataFrame(train[0]), 
                                         img_size, pca, batch_size=batch_size)
    
    print("\n=== Processing Test Data ===")
    # Transform test data using the same PCA
    X_test = transform_features_batched(images_folder_path, pd.DataFrame(test[0]), 
                                        img_size, pca, batch_size=batch_size)

    Y_train, mlb = encode_predictor_labels(train[1])
    Y_test, _ = encode_predictor_labels(test[1])

    X_train_encoded = encode_scale_input_features(pd.DataFrame(train[0]), X_train)
    X_test_encoded = encode_scale_input_features(pd.DataFrame(test[0]), X_test)

    # Save PCA model for later use
    pca_model_path = os.path.join(DATA_DIRECTORY_PATH, "pca_model.pkl")
    with open(pca_model_path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"\nPCA model saved to {pca_model_path}")

    print("\n=== Training KNN Model ===")
    knn_model = train_knn_model(X_train_encoded, Y_train, n_neighbors=5)
    accuracy = evaluate_model(knn_model, X_test_encoded, Y_test)
    print(f"KNN Model Accuracy: {accuracy * 100:.2f}%")
    
    # Save KNN model
    knn_model_path = os.path.join(DATA_DIRECTORY_PATH, "knn_model.pkl")
    with open(knn_model_path, 'wb') as f:
        pickle.dump(knn_model, f)
    print(f"KNN model saved to {knn_model_path}")
    