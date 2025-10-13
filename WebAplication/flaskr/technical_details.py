from flask import Blueprint, render_template

bp = Blueprint('technical_details', __name__)


@bp.route('/technical_details')
def index():
    return render_template('technical_details/technical_details.html')