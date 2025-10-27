# Challenges and Solutions

## Preprocessing Pipeline Simplification

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
