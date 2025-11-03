# Challenges and Solutions

## Preprocessing Pipeline Simplification
### first preprocessing pipeline
The initial preprocessing pipeline for the X-ray images involved several complex steps aimed at enhancing soft tissue visibility. This included multiple layers of histogram equalization, contrast adjustments, and noise reduction techniques. While these steps improved the visual quality of the images for human interpretation, they inadvertently complicated the feature extraction process for the KNN classifier.


### What the New `preprocess_xray` Does

The new `preprocess_xray` function implements a simplified, standard preprocessing pipeline for X-ray images with three core steps:

1. **HSV Color Space Conversion** - Converts the image from BGR color space to HSV (Hue, Saturation, Value). This separates the brightness information (V channel) from color information, allowing us to enhance contrast without affecting color properties.

2. **CLAHE on V Channel** - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) specifically to the Value (brightness) channel. This is a smarter version of histogram equalization that works on small regions (8x8 tiles) of the image instead of the whole image at once. This prevents over-brightening already bright areas while still enhancing darker regions. The "contrast limited" part (clipLimit=2.0) prevents noise from being amplified too much.

3. **Resizing** - Converts back to grayscale and normalizes all images to a consistent size (2500x2048) for uniform processing

### Why We Changed It

The previous preprocessing pipeline was overly complex with multiple enhancement layers optimized for soft tissue visibility. While this made images easier for human comparison, it actually hindered the KNN's ability to extract meaningful features for classification.

The issue was that aggressive preprocessing can distort the underlying patterns that machine learning algorithms rely on. By switching to this simpler pipeline, we can:

- **Start with a baseline** that performs minimal transformations
- **Incrementally test** which preprocessing steps actually improve KNN performance
- **Identify which techniques help vs. hurt** feature extraction

This iterative approach allows us to build an evidence-based pipeline tailored specifically to what the KNN model needs, rather than what looks good to human eyes.


# add edges and sift features
to pass further information to the following processing stage we extract edges using Canny edge detection and SIFT features from the preprocessed X-ray images. This additional information can help improve the performance of the KNN classifier by providing more distinctive features for classification.

adding these into a png was not possible so we save them in a npy file

is adding the sift features really reasonable in the preprocessing stage? storing it takes a significant amount of space and the sift features depend on parameters that might be better tuned later in the pipeline.