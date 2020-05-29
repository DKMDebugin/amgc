import numpy
import pandas
import sys
MODULE_PATH = '/home/macbookretina/automatic-music-genre-classification/feature_extraction_deep_learning'
sys.path.insert(1, MODULE_PATH)
from custom_module.utilities import *

def extract_from_gtzan():
    # extract log-mel / constant-Q transform and mel-spectomgram in gtzan

    # create an empty list to store extract feature and label
    path = '/home/macbookretina/s3-bucket/data'
    cqts = []
    mel_spects = []
    genre_labels = []
    data_sources = []

    print('extracting log-mel and mel-spectogram from gtzan dataset')

    for file in os.scandir(MOUNTED_DATASET_PATH + '/gtzan/wavfiles'):
        if file.is_file():
            # extract genre label
            genre_label = str(file.name).split('.')[0]

            if genre_label in GENRES:
                scaled_cqt = extract_cqt(file);
                scaled_mel_spect = extract_mel_spect(file);

                # adjust shape to the shape with most occurence
                if scaled_cqt.shape[1] != 2812:
                    scaled_cqt.resize(84, 2812, refcheck=False)

                if scaled_mel_spect.shape[1] != 2812:
                    scaled_mel_spect.resize(128, 2812, refcheck=False)

                # append to list
                genre_labels.append(genre_label)
                # flatten to fit into dataframe and add to the list
                scaled_cqt = scaled_cqt.flatten()
                cqts.append(scaled_cqt)
                scaled_mel_spect = scaled_mel_spect.flatten()
                mel_spects.append(scaled_mel_spect)

                feedback(file, genre_label)

    # check if all the lists are equal in length and throw an exception if not
    print('checking if the lists are equal in size.')
    is_all_equal_in_length = len(cqts) == len(mel_spects) == len(genre_labels)
    assert (is_all_equal_in_length), \
                'cqts: ' + str(len(cqts)) + \
                ' mel_spects: ' + str(len(mel_spects)) + \
                ' genre_labels: ' + str(len(genre_labels))

    # convert the lists to arrays so it can be stacked
    print('converting lists to numpy array')
    cqts = numpy.array(cqts)
    mel_spects = numpy.array(mel_spects)
    length = len(cqts)
    genre_labels = numpy.array(genre_labels).reshape(length, 1)
    data_sources = numpy.array(['gtzan']*length).reshape(length, 1)

    # create dataframes and save as csvs
    print('stacking array & saving as csv')
    cqt_df = pandas.DataFrame(numpy.hstack((genre_labels, data_sources, cqts)))
    mel_spect_df = pandas.DataFrame(numpy.hstack((genre_labels, data_sources, mel_spects)))
    cqt_df.to_csv(path + '/cqt_gtzan.csv')
    mel_spect_df.to_csv(path + '/mel_spect_gtzan.csv')
    print('done')


if __name__ == '__main__':
    extract_from_gtzan()