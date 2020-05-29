import numpy
from audioread import NoBackendError
import pandas
import sys
MODULE_PATH = '/home/macbookretina/automatic-music-genre-classification/feature_extraction_deep_learning'
sys.path.insert(1, MODULE_PATH)
from custom_module.utilities import *

# def process_and_save(cqts, mel_spects, genre_labels, count):
#     # check if all the lists are equal in length and throw an exception if not
#     path = '/home/macbookretina/s3-bucket/data'
#     count = str(count)
#     print('checking if the lists are equal in size.')
#     is_all_equal_in_length = len(cqts) == len(mel_spects) == len(genre_labels)
#     assert (is_all_equal_in_length), \
#                 'cqts: ' + str(len(cqts)) + \
#                 ' mel_spects: ' + str(len(mel_spects)) + \
#                 ' genre_labels: ' + str(len(genre_labels))

#     # convert the lists to arrays so it can be stacked
#     print('converting lists to numpy array')
#     cqts = numpy.array(cqts)
#     mel_spects = numpy.array(mel_spects)
#     length = len(cqts)
#     genre_labels = numpy.array(genre_labels).reshape(length, 1)
#     data_sources = numpy.array(['fma']*length).reshape(length, 1)
#     data_cqts = {
#         'genre_labels': genre_labels,
#         'data_sources': data_sources,
#         'cqts': cqts
#     }
#     data_mel = {
#         'genre_labels': genre_labels,
#         'data_sources': data_sources,
#         'cqts': cqts
#     }

#     # create dataframes and save as csvs
#     print('stacking array & saving as csv')
# #     cqt_df = pandas.DataFrame(numpy.hstack((genre_labels, data_sources, cqts)))
# #     mel_spect_df = pandas.DataFrame(numpy.hstack((genre_labels, data_sources, mel_spects)))
#     cqt_df = pandas.DataFrame(data_cqts)
#     mel_spect_df = pandas.DataFrame(data_mel)
#     cqt_df.to_csv(path + '/cqt_fma_' + count + '.csv')
#     mel_spect_df.to_csv(path + '/mel_spect_fma_' + count + '.csv')
#     print('saved batch: ' + count + '!!')
#     return [], [], []

def extract_from_fma():
    # extract log-mel / constant-Q transform and mel-spectomgram in fma
    
    # collect track id and genres of tracks in the small subset.
    print('collecting track id and genres of tracks in the small subset of fma dataset')
    tracks = load(MOUNTED_DATASET_PATH + '/fma_metadata/tracks.csv')
    fma_full = tracks[[('set', 'subset'), ('track', 'genre_top')]]
    small_subset = fma_full[('set', 'subset')] == 'small'
    fma_small = fma_full[small_subset]
    fma_small = pd.DataFrame({
        'subset': fma_small[('set', 'subset')],
        'label': fma_small[('track', 'genre_top')]
    })
    print('collected track id and genres of tracks in the small subset of fma')

    # create an empty list to store extract feature and label
    cqts = []
    mel_spects = []
    genre_labels = []
    data_sources = []
    print('extracting log-mel and mel-spectogram from fma dataset')
    count = 0
    for directory in os.scandir(MOUNTED_DATASET_PATH + '/fma_small'):
        if directory.is_dir():
            for file in os.scandir(directory.path):
#                 if count == 1:
#                     cqts, mel_spects, genre_labels = process_and_save(
#                         cqts, mel_spects, genre_labels, count)
                    
                if file.is_file():
                    # extract track id and map track id to genre label
                    track_id = int(file.name[:-4].lstrip('0'))
                    genre_label = fma_small.at[track_id, 'label'].lower().replace('-', '')

                    if genre_label in GENRES:
                        try:
                            scaled_cqt = extract_cqt(file);
                            scaled_mel_spect = extract_mel_spect(file);
                        except RuntimeError:
                            print('RuntimeError')
                            continue
                        except NoBackendError:
                            print('NoBackendError')
                            continue

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
            count = count + 1

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
    data_sources = numpy.array(['fma']*length).reshape(length, 1)

#     # create dataframes and save as csvs
#     print('stacking array & saving as csv')
#     cqt_df = pandas.DataFrame(numpy.hstack((genre_labels, data_sources, cqts)))
#     mel_spect_df = pandas.DataFrame(numpy.hstack((genre_labels, data_sources, mel_spects)))
#     cqt_df.to_csv(path + '/cqt_fma.csv')
#     mel_spect_df.to_csv(path + '/mel_spect_fma.csv')
#     print('done')

if __name__ == '__main__':
    extract_from_fma()