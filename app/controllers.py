from flask import Flask
from flask import render_template,request

import numpy as np

app = Flask(__name__)

@app.route('/')
def hello_world(greeting='Hello, World!'):
    return render_template('index.html', greeting=greeting)

@app.route('/audio', methods=['POST'])
def store_audio():
    if request.method == 'POST':
        file = request.files['audio_file']
        file.save('audio_file.mp3')
        # data = request.form['array_buffer']
        # print(data)
        # # audio_np = np.frombuffer(data)
        # # print(audio_np)
        return '200'

if __name__ == '__main__':
    app.run()
