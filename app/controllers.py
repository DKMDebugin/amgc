import json
import os

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

import sys

# REMOTE_MODULE_PATH = '/home/macbookretina/automatic-music-genre-classification/feature_extraction_deep_learning'
LOCAL_MODULE_PATH = '/Users/macbookretina/Desktop/automatic-music-genre-classification/feature_extraction_deep_learning'
sys.path.insert(1, LOCAL_MODULE_PATH)
from custom_module.utilities import extract_features_make_prediction

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = f'{WORKING_DIR}/file'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename=''):
    return '.' in filename and \
           filename.split('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world(greeting='Hello, World!'):
    return render_template('index.html', greeting=greeting)


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
                print(prediction)
                return jsonify(prediction), 200

        return '400'


if __name__ == '__main__':
    app.run(debug=True)

# first extract features
# scale features
# make predictions
# return result
