# AMGC: Automatic Music Genre Classification
> This is repository was created by **Jamiu Olashile Salimon** for his thesis titled: **"The Comparison of Different Music Features for Automatic Music Genre Classification using Deep Learning"** submitted to the University of Limerick, August 2020 as a partial fulfillment of the requirements for the degree of Master of Science in Software Engineering.

A system was created around the final model from the experiment. The core operation carried out by the system is as follows;
* Live capturing of music audio (max 30s).
* Automatically stop recording after 30s
* Detect silence
* Extract music features from audio signal.
* Compute stats repsentation & scale features.
* Correctly classify captured music audio by genre
* Present result to user in a pie chart.


![Project walk through](assets/app.gif)

## Development & Installation Setup

To setup, you need to clone or download the master branch of this repository then run the following commands in the base directory of the project;
```sh
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Test your set-up by running the project test suites with the command below;
```sh
python -m unittest amgc_project/tests/test_prediction.py
```

After all the test suites passes, you can then run the app with the command below;
```sh
python amgc_project/app/__init__.py
```

Click on [http://127.0.0.1:5000](http://127.0.0.1:5000) to access the UI locally  on your favourite browser.

## Release History
* 2.0.0
    * Version 2
* 1.0.0
    * Version 1

## Meta

Jamiu Olashile Salimon – [@DKMDebugin](https://www.linkedin.com/in/dkmdebugin/) – salimonjamiu96@gmail.com

Distributed under the MIT license. See ``LICENSE.txt`` for more information.

[https://github.com/DKMDebugin/amgc](https://github.com/DKMDebugin/amgc)

## Contributing

1. Fork it (<https://github.com/DKMDebugin/amgc/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

