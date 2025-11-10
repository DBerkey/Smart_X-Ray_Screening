"""
User Guide module for the Smart X-Ray Screening web application.

This module serves the user guide section of the application, providing:
- Instructions on acceptable image types (JPG, JPEG, PNG, BMP)
- Step-by-step guidance for uploading X-ray images
- Detailed explanation of how to interpret analysis results
- Information about image requirements and limitations
- Instructions for using the PDF export feature

The user guide is designed to help users effectively use the X-ray
screening system and understand their analysis results.
"""

from flask import Blueprint, render_template

# Blueprint for user guide page routes
bp = Blueprint('user_guide', __name__)


@bp.route('/user_guide')
def index():
    """
    Render the user guide page.

    Serves a comprehensive guide that includes:
    - Accepted image formats (JPG, JPEG, PNG, BMP)
    - Step-by-step upload instructions
    - Patient data input requirements
    - Guide to interpreting results
    - Information about the results page components
    - Export and printing options

    The page includes animated GIFs demonstrating the upload and
    analysis process for better user understanding.

    Returns:
        str: Rendered HTML content of the user guide template.
    """
    return render_template('user_guide/user_guide.html')