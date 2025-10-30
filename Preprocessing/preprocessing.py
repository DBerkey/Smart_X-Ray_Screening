"""
This module contains functions to preprocess X-Ray images for further analysis with a KNN.
Copilot was used to assist in writing this code.
"""

import cv2
import numpy as np
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

input_directory = 'images'
output_directory = 'preprocessed'
NUM_THREADS = 4


def brightness_range_maximisation(image):
    """
    make the darkest pixel 0 and the brightest pixel 255
    
    Args:
        image (numpy.ndarray): Input grayscale image
    Returns:
        numpy.ndarray: Brightness adjusted image
    """

    min_val = np.min(image)  # Keep min from entire image
    max_val = np.max(image)  # Keep max from entire image
    
    # Scale pixel values to the full 0-255 range
    scaled = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return scaled


def detect_edges(image):
    """
    Detect edges in the image using Canny edge detection.
    
    Args:
        image (numpy.ndarray): Input grayscale image
    Returns:
        numpy.ndarray: Edge-detected image as binary mask
    """
    # blurring
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    blurred = cv2.medianBlur(blurred, 5)


    # Apply Canny edge detection
    threshold1 = 25
    edges = cv2.Canny(blurred, threshold1=threshold1, threshold2=3*threshold1, apertureSize=3)
    
    return edges


def build_output_data(images):
    """
    Build output data structure.
    
    Args:
        images (list): List of preprocessed images
    Returns:
        output image
    """
    # stack images along a new dimension
    output = np.stack(images, axis=-1)
    return output


def preprocess_xray(image_path, output_size=(2500, 2048), show_process=False, process_types=['standard']):
    """
    Apply standard preprocessing pipeline to an X-ray image.
    
    Args:
        image_path (str): Path to the input X-ray image
        output_size (tuple): Target size for the output image (width, height)
        show_process (bool): Whether to save intermediate processing steps
        process_types (list): List of processing types to apply (currently only 'standard' is implemented)
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    # Image processing log
    workbench = []
    processed = []
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    workbench.append(image)

    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert to HSV (load as color first)
    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    workbench.append(hsv_image)

    # Apply CLAHE to V channel
    h, s, v = cv2.split(hsv_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)
    hsv_clahe = cv2.merge([h, s, v_clahe])
    workbench.append(hsv_clahe)

    # Convert back to grayscale for consistency
    gray_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    gray_clahe = cv2.cvtColor(gray_clahe, cv2.COLOR_BGR2GRAY)
    workbench.append(gray_clahe)

    # Resize to target size
    resized = cv2.resize(workbench[-1], output_size)
    workbench.append(resized)
    # Record processed image if required
    if 'standard' in process_types:
        processed.append(workbench[-1])

    # detect edges
    edges = detect_edges(workbench[-1])
    workbench.append(edges)
    # Record processed image if required
    if 'edges' in process_types:
        processed.append(edges)


    # build output data
    output_data = build_output_data(processed)


    if show_process:
        # Clear and create pipeline directory
        pipeline_dir = 'standard_pipeline'
        if os.path.exists(pipeline_dir):
            shutil.rmtree(pipeline_dir)
        os.makedirs(pipeline_dir)
        
        for i, img in enumerate(workbench):
            output_path = os.path.join(pipeline_dir, f'{i:02d}.png')
            cv2.imwrite(output_path, img)
            print(f"Saved standard pipeline stage: {output_path}")

    return output_data


def preprocess_batch(input_dir, output_dir, output_size=(2500, 2048), num_threads=NUM_THREADS, process_types=['standard']):
    """
    Preprocess all X-ray images in a directory with soft tissue optimization and save them to an output directory.
    Uses multithreading for improved performance.
    
    Args:
        input_dir (str): Path to the directory containing input images
        output_dir (str): Path to the directory where processed images will be saved
        output_size (tuple): Target size for the output images (width, height)
        num_threads (int): Number of threads to use for parallel processing
    """
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    filepaths = []
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            input_path = os.path.join(input_dir, filename)
            filepaths.append(input_path)
    
    # Use the multithreaded function to process all images
    preprocess_filepaths_threaded(filepaths, output_dir, num_threads, output_size, process_types)


def preprocess_filepaths_threaded(filepaths, output_dir, num_threads=NUM_THREADS, output_size=(2500, 2048), process_types=['standard']):
    """
    Preprocess a list of image filepaths using multiple threads.
    
    Args:
        filepaths (list): List of paths to input images
        output_dir (str): Path to the directory where processed images will be saved
        num_threads (int): Number of threads to use for parallel processing
        output_size (tuple): Target size for the output images (width, height)
        process_types (list): List of processing types to apply
    
    Returns:
        tuple: (successful_count, error_count)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    def process_single_file(filepath):
        try:
            filename = os.path.basename(filepath)
            base_name = os.path.splitext(filename)[0]
            processed_image = preprocess_xray(filepath, output_size, process_types=process_types)
            
            # Check if multi-channel output
            if len(processed_image.shape) == 3 and processed_image.shape[2] > 1:
                # Save as numpy file for multi-channel data
                output_path = os.path.join(output_dir, f"processed_{base_name}.npy")
                np.save(output_path, processed_image)
            else:
                # Save as image for single channel
                output_path = os.path.join(output_dir, f"processed_{filename}")
                # Handle single channel stored as 3D array
                if len(processed_image.shape) == 3:
                    processed_image = processed_image[:, :, 0]
                cv2.imwrite(output_path, processed_image)
            return True, filepath
        except Exception as e:
            return False, f"{filepath}: {str(e)}"
    
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_single_file, filepaths))
    
    successful = sum(1 for success, _ in results if success)
    errors = sum(1 for success, _ in results if not success)
    
    print(f"\nThreaded processing complete:")
    print(f"Successfully processed: {successful} images")
    print(f"Errors encountered: {errors} images")
    
    return successful, errors


if __name__ == "__main__":
    # Example usage:

    #uncomment to test threaded batch processing
    #images = [
    #    'images/00000372_008.png',
    #    'images/00000001_000.png',
    #    'images/00000032_005.png'
    #]
    #output_directory = 'preprocessed_threaded'
    #preprocess_filepaths_threaded(images, output_directory,3,(2500, 2048))

    #uncomment to test batch processing
    preprocess_batch(input_directory, output_directory, process_types=['standard','edges'])

    #uncommment to test single image processing with process visualization
    preprocess_xray('images/00000001_000.png', show_process=True)