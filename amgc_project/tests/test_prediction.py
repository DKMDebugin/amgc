import os
import unittest
from amgc_project.app import app


class TestPrediction(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True
        self.filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], 'audio.wav')

    def test_prediction_status_code(self):
        response = self.client.post('/audio', data={
            'audio_data': open(self.filepath, "rb")
        })
        print(response.get_json())
        assert response.status_code == 200

    def test_prediction_response_datatype_len(self):
        response = self.client.post('/audio', data={
            'audio_data': open(self.filepath, "rb")
        })
        data = response.get_json()
        assert type(data) == dict
        assert len(data) == 2

    def test_prediction_file_ext(self):
        mp3_filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], 'audio.txt')
        response = self.client.post('/audio', data={
            'audio_data': open(mp3_filepath, 'r')
        })
        assert response.status_code == 400


if __name__ == '__main__':
    unittest.main()