"""
This module contains functions to preprocess X-Ray images for further analysis with a KNN.
"""

import cv2
import numpy as np
import os
import shutil


def brightness_range_maximisation(image):
    """
    make the darkest pixel 0 and the brightest pixel 255
    
    Args:
        image (numpy.ndarray): Input grayscale image
    Returns:
        numpy.ndarray: Brightness adjusted image
    """

    # Define region of interest (ROI) - adjust these coordinates as needed
    height, width = image.shape
    roi_y1, roi_y2 = height // 4, 3 * height // 4  # Middle 50% vertically
    roi_x1, roi_x2 = width // 4, 3 * width // 4    # Middle 50% horizontally
    
    # Extract the region of interest
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Find min and max values in the ROI
    min_val = np.min(roi)
    max_val = np.max(image)  # Keep max from entire image
    
    # Scale pixel values to the range [0, 255] using float division
    adjusted = 255.0 * np.maximum((image - min_val),0) / np.maximum((max_val - min_val),1)
    
    return adjusted.astype(np.uint8)


def enhance_visibility(image):
    """
    """
    workbench = [image]
    # Apply median blur to reduce noise
    blurred = cv2.medianBlur(workbench[-1], 5)
    workbench.append(blurred)

    # subtract blurred image from original to enhance edges
    diff = cv2.subtract(workbench[-2], workbench[-1])
    workbench.append(diff)

    #add diff to original to enhance visibility
    enhanced = cv2.add(workbench[-3], workbench[-1])
    workbench.append(enhanced)

    #apply clahe to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(workbench[-1])
    workbench.append(clahe_enhanced)

    #maximise brightness range
    brightness_maximised = brightness_range_maximisation(workbench[-1])
    workbench.append(brightness_maximised)

    #median blur again to reduce noise
    workbench.append(cv2.medianBlur(workbench[-1], 5))

    return workbench[-1]


def preprocess_xray(image_path, output_size=(2500, 2048), show_process=False):
    """
    Apply the complete preprocessing pipeline to an X-ray image.
    
    Args:
        image_path (str): Path to the input X-ray image
        output_size (tuple): Target size for the output image (width, height)
    
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    #image processing log
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

    # Enhance visibility
    enhanced = enhance_visibility(workbench[-1])
    workbench.append(enhanced)
    
    if show_process:
        # Clear and create pipeline directory
        pipeline_dir = 'pipeline'
        if os.path.exists(pipeline_dir):
            shutil.rmtree(pipeline_dir)
        os.makedirs(pipeline_dir)

        # Save all workbench images with descriptive names
        for i, img in enumerate(workbench):
            output_path = os.path.join(pipeline_dir, f'{i:02d}.png')
            cv2.imwrite(output_path, img)
            print(f"Saved pipeline stage: {output_path}")

    return workbench[-1]

def preprocess_batch(input_dir, output_dir, output_size=(2500, 2048)):
    """
    Preprocess all X-ray images in a directory and save them to an output directory.
    
    Args:
        input_dir (str): Path to the directory containing input images
        output_dir (str): Path to the directory where processed images will be saved
        output_size (tuple): Target size for the output images (width, height)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # Preprocess the image
                processed_image = preprocess_xray(input_path, output_size)
                
                # Save the processed image
                cv2.imwrite(output_path, processed_image)
                print(f"Processed: {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")



input_directory = 'images'
output_directory = 'preprocessed'
preprocess_batch(input_directory, output_directory)

img_path = 'images/00000372_008.png'
preprocess_xray(img_path, show_process=True)