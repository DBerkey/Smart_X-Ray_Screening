"""
This module contains functions to preprocess X-Ray images for further analysis with a KNN.
Copilot was used to assist in writing this code.
"""

import cv2
import numpy as np
import os
import shutil

input_directory = 'images'
output_directory = 'preprocessed'
input_of_single_image = 'images/00000372_008.png' #for testing single image processing with process visualization


def soft_tissue_contrast_enhancement(image):
    """
    Enhance contrast specifically for soft tissue structures.
    Uses low clip limit CLAHE optimized for soft tissue visibility.
    
    Args:
        image (numpy.ndarray): Input grayscale image
    Returns:
        numpy.ndarray: Soft tissue enhanced image
    """
    # Use smaller tile size and lower clip limit for soft tissue
    clahe_soft = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(6, 6))
    enhanced = clahe_soft.apply(image)
    
    # Apply gentle gamma correction to enhance darker regions (soft tissue)
    gamma = 0.8
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(enhanced, table)
    
    return gamma_corrected


def lung_field_enhancement(image):
    """
    Specifically enhance lung fields for pneumonia/infiltrate detection.
    
    Args:
        image (numpy.ndarray): Input grayscale image
    Returns:
        numpy.ndarray: Lung field enhanced image
    """
    # Create a mask for lung regions (typically mid-intensity areas)
    lung_mask = cv2.inRange(image, 30, 150)
    lung_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_CLOSE, 
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    
    # Apply specific enhancement to lung regions
    clahe_lung = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lung_enhanced = clahe_lung.apply(image)
    
    # Blend enhanced lung regions with original image
    result = image.copy()
    result[lung_mask > 0] = lung_enhanced[lung_mask > 0]
    
    return result


def cardiac_silhouette_enhancement(image):
    """
    Enhance cardiac silhouette and mediastinal structures.
    
    Args:
        image (numpy.ndarray): Input grayscale image
    Returns:
        numpy.ndarray: Cardiac enhanced image
    """
    # Use bilateral filter to preserve edges while smoothing
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Apply edge-preserving contrast enhancement
    clahe_cardiac = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(10, 10))
    enhanced = clahe_cardiac.apply(bilateral)
    
    # Enhance using weighted combination
    result = cv2.addWeighted(image, 0.6, enhanced, 0.4, 0)
    
    return result


def soft_tissue_edge_enhancement(image):
    """
    Enhance edges in soft tissue while preserving smooth gradients.
    
    Args:
        image (numpy.ndarray): Input grayscale image
    Returns:
        numpy.ndarray: Edge enhanced image
    """
    # Apply Gaussian blur with different sigmas
    blur1 = cv2.GaussianBlur(image, (0, 0), 1.0)
    blur2 = cv2.GaussianBlur(image, (0, 0), 2.0)
    
    # Create difference of Gaussians for edge enhancement
    dog = cv2.subtract(blur1, blur2)
    
    # Add back to original with controlled strength
    enhanced = cv2.addWeighted(image, 1.0, dog, 0.5, 0)
    
    # Apply median filter to reduce noise
    denoised = cv2.medianBlur(enhanced, 3)
    
    return denoised


def soft_tissue_tumor_enhancement(image):
    """
    Enhance visibility of soft tissue masses and tumors.
    
    Args:
        image (numpy.ndarray): Input grayscale image
    Returns:
        numpy.ndarray: Tumor-enhanced image
    """
    # Use morphological operations to enhance rounded structures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Top-hat transform to highlight bright masses
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    
    # Black-hat transform to highlight dark masses
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    
    # Combine with original image
    enhanced = cv2.add(image, tophat)
    enhanced = cv2.subtract(enhanced, blackhat)
    
    # Apply adaptive contrast enhancement
    clahe_tumor = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    final = clahe_tumor.apply(enhanced)
    
    return final


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


def enhance_soft_tissue_visibility(image):
    """
    Comprehensive soft tissue enhancement pipeline combining multiple techniques.
    Optimized for pneumonia, cardiac abnormalities, and soft tissue masses.
    
    Args:
        image (numpy.ndarray): Input grayscale image
    Returns:
        numpy.ndarray: Soft tissue enhanced image
    """
    workbench = [image]
    
    #Initial soft tissue contrast enhancement
    soft_contrast = soft_tissue_contrast_enhancement(workbench[-1])
    workbench.append(soft_contrast)
    
    #Enhance lung fields specifically
    lung_enhanced = lung_field_enhancement(workbench[-1])
    workbench.append(lung_enhanced)

    #Enhance cardiac silhouette
    cardiac_enhanced = cardiac_silhouette_enhancement(workbench[-1])
    workbench.append(cardiac_enhanced)
    
    #Apply soft tissue edge enhancement
    edge_enhanced = soft_tissue_edge_enhancement(workbench[-1])
    workbench.append(edge_enhanced)

    #apply gausian blur to reduce noise
    blurred = cv2.GaussianBlur(workbench[-1], (5, 5), 0)
    workbench.append(blurred)
    
    #Final brightness optimization
    final_enhanced = brightness_range_maximisation(workbench[-1])
    workbench.append(final_enhanced)
    
    return workbench[-1]


def preprocess_xray_soft_tissue(image_path, output_size=(2500, 2048), show_process=False):
    """
    Apply soft tissue optimized preprocessing pipeline to an X-ray image.
    
    Args:
        image_path (str): Path to the input X-ray image
        output_size (tuple): Target size for the output image (width, height)
        show_process (bool): Whether to save intermediate processing steps
    
    Returns:
        numpy.ndarray: Preprocessed image array optimized for soft tissue visibility
    """
    # Image processing log
    workbench = []
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    workbench.append(image)
    
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Resize image to target size
    resized = cv2.resize(workbench[-1], output_size)
    workbench.append(resized)

    # Apply brightness adjustment
    brightened = brightness_range_maximisation(workbench[-1])
    workbench.append(brightened)

    # Apply soft tissue specific enhancement
    soft_tissue_enhanced = enhance_soft_tissue_visibility(workbench[-1])
    workbench.append(soft_tissue_enhanced)
    
    if show_process:
        # Clear and create pipeline directory
        pipeline_dir = 'soft_tissue_pipeline'
        if os.path.exists(pipeline_dir):
            shutil.rmtree(pipeline_dir)
        os.makedirs(pipeline_dir)

        # Save all workbench images with descriptive names
        stage_names = [
            'original',
            'resized',
            'brightness_adjusted',
            'soft_tissue_enhanced'
        ]
        
        for i, (img, name) in enumerate(zip(workbench, stage_names)):
            output_path = os.path.join(pipeline_dir, f'{i:02d}_{name}.png')
            cv2.imwrite(output_path, img)
            print(f"Saved soft tissue pipeline stage: {output_path}")

    return workbench[-1]


def preprocess_batch_soft_tissue(input_dir, output_dir, output_size=(2500, 2048)):
    """
    Preprocess all X-ray images in a directory with soft tissue optimization and save them to an output directory.
    
    Args:
        input_dir (str): Path to the directory containing input images
        output_dir (str): Path to the directory where processed images will be saved
        output_size (tuple): Target size for the output images (width, height)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    processed_count = 0
    error_count = 0
    
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"soft_tissue_{filename}")
            
            try:
                # Preprocess the image with soft tissue optimization
                processed_image = preprocess_xray_soft_tissue(input_path, output_size)
                
                # Save the processed image
                cv2.imwrite(output_path, processed_image)
                print(f"Soft tissue processed: {filename}")
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                error_count += 1
    
    print(f"\nSoft tissue processing complete:")
    print(f"Successfully processed: {processed_count} images")
    print(f"Errors encountered: {error_count} images")


preprocess_batch_soft_tissue(input_directory, output_directory)

#uncommment to test single image processing with process visualization
preprocess_xray_soft_tissue(input_of_single_image, show_process=True)