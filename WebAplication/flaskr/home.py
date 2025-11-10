"""
Home page blueprint for the Smart X-Ray Screening web application.

This module defines the routes and handlers for the application's home page.
It provides the main entry point for users to access the X-ray screening system.
"""

from flask import Blueprint, render_template

# Blueprint for the home page routes
bp = Blueprint('home', __name__)


@bp.route('/')
def index():
    """
    Render the application's home page.

    This is the main entry point for the Smart X-Ray Screening application.
    Serves as the landing page where users can start their X-ray analysis journey.

    Returns:
        str: Rendered HTML content of the home page template.
    """
    return render_template('home/home.html')