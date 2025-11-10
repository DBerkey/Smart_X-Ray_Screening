"""
Technical Details module for the Smart X-Ray Screening web application.

This module serves the technical documentation section of the application,
providing comprehensive information about:
- System requirements (storage, RAM, CPU specifications)
- Data processing pipeline (HSV conversion, CLAHE, edge detection)
- Two-stage machine learning architecture:
  * Stage 1: Binary SVM classifier for abnormality detection
  * Stage 2: Multi-label disease classification
- Performance metrics and model evaluation results
- Preprocessing techniques and image enhancement methods

The technical documentation helps developers and technical users understand
the system's architecture, requirements, and implementation details.
"""

from flask import Blueprint, render_template

# Blueprint for technical details page routes
bp = Blueprint('technical_details', __name__)


@bp.route('/technical_details')
def index():
    """
    Render the technical details page.

    Serves detailed technical documentation including:
    - System Requirements:
        * 200GB storage requirement
        * 32GB DDR5-6000 RAM specifications
        * AMD Ryzen 7 7700X processor requirements
    - Data Processing Pipeline:
        * HSV color space conversion
        * Contrast Limited Adaptive Histogram Equalization (CLAHE)
        * Grayscale normalization
        * Canny edge detection
    - Machine Learning Architecture:
        * Stage 1: Binary SVM with 69.13% accuracy
        * Stage 2: Multi-label disease classification
        * Performance metrics (F1 scores, accuracy, hamming loss)
    - Links to GitHub repository for full implementation details

    Returns:
        str: Rendered HTML content of the technical details template.
    """
    return render_template('technical_details/technical_details.html')