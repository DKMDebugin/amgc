from amgc_project.app.create_amgc_app import *

app = create_amgc_app()

if __name__ == '__main__':
    # set flask cmd variables
    command = 'export FLASK_APP=app;export FLASK_ENV=development'
    os.system(command)
    # run app
    app.run(debug=True)