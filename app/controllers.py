import os
import sys

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.dirname(WORKING_DIR)
sys.path.insert(1, BASE_DIR + '/feature_extraction_deep_learning/')

from custom_module.utilities import extract_features_make_prediction

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
