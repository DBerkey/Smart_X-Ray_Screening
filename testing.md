## Preprocessing Testing

### Visual Verification Process
- Used `show_process=True` flag to generate intermediate processing stages
- Examined each step with sample image (`00000001_000.png`) saved to `standard_pipeline/` directory
- Verified processing stages:
  - **00.png**: Original grayscale image loaded correctly
  - **01.png**: HSV color space conversion preserved image structure
  - **02.png**: CLAHE applied to V channel enhanced contrast appropriately
  - **03.png**: Grayscale conversion maintained enhanced details
  - **04.png**: Resized to target dimensions (2500×2048) without distortion
  - **05.png**: Edge detection (Canny) identified relevant boundaries
- Confirmed final multi-channel output displayed correctly with `cv2.imshow()`
- Validated both 'standard' and 'edges' channels contained expected visual information