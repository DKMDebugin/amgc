import os
import sys

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(WORKING_DIR))
sys.path.insert(1, BASE_DIR + '/feature_extraction_deep_learning/')

from custom_module.utilities import extract_features_make_prediction


def create_amgc_app():
    """
    create_app() creates a flask app.
    """
    UPLOAD_FOLDER = f'{WORKING_DIR}/file'
    ALLOWED_EXTENSIONS = {'wav'}

    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(SECRET_KEY='dev')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # utility
    def allowed_file(filename=''):
        return '.' in filename and \
               filename.split('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    # routes & controllers
    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/audio', methods=['POST'])
    def store_audio():
        if request.method == 'POST':
            file = request.files.get('audio_data')
            if file.filename != '':
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    prediction = extract_features_make_prediction(filepath)
                    return jsonify(prediction), 200

                return {'Error': 'File is not in .wav format.'}, 400

            return {'Error': 'File has no filename'}, 400

    return app
