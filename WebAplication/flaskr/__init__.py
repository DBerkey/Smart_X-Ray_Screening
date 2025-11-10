"""
Flask application factory module for the Smart X-Ray Screening web application.

This module initializes and configures the Flask application, sets up configuration,
and registers all the blueprints for different parts of the application including
home, user guide, technical details, results, and analysis pages.
"""

import os

from flask import Flask
from . import home, user_guide, technical_details, results, analyze


def create_app(test_config=None):
    """
    Application factory function that creates and configures a Flask application instance.

    Args:
        test_config (dict, optional): Configuration dictionary for testing. 
            When None, the app uses the default configuration. Defaults to None.

    Returns:
        Flask: A configured Flask application instance.

    The function performs the following tasks:
    1. Creates a Flask application instance
    2. Sets basic configuration including secret key and database path
    3. Loads additional configuration from config.py if available
    4. Ensures instance folder exists
    5. Registers all application blueprints
    """
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # register blueprints
    app.register_blueprint(home.bp)
    app.register_blueprint(user_guide.bp)
    app.register_blueprint(technical_details.bp)
    app.register_blueprint(results.bp)
    app.register_blueprint(analyze.bp)

    return app
