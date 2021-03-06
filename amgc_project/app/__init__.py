import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, ROOT_DIR)

from app.create_amgc_app import create_amgc_app

app = create_amgc_app()

if __name__ == '__main__':
    # set flask cmd variables
    command = 'export FLASK_APP=app;export FLASK_ENV=development'
    os.system(command)
    # run app
    app.run(debug=True)