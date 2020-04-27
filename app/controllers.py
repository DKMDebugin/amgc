import os

from flask import Flask
from flask import render_template, request

from werkzeug.utils import secure_filename

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
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return '200'

        return '400'

if __name__ == '__main__':
    app.run(debug=True)


