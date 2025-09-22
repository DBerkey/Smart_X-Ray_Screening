from flask import Blueprint, render_template

bp = Blueprint('user_guide', __name__)


@bp.route('/user_guide')
def index():
    return render_template('user_guide/user_guide.html')