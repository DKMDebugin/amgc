import ast
from pandas.api.types import CategoricalDtype
import librosa
import numpy as np
import os
import pandas as pd
import pywt
from scipy.stats import skew

# constants

MOUNTED_DATASET_PATH = '/home/macbookretina/s3-bucket'
LOCAL_MOUNTED_DATASET_PATH = '/Users/macbookretina/Desktop/mount-s3-bucket'
SAMPLE_FILE = '/home/macbookretina/s3-bucket/gtzan/wavfiles/blues.00042.wav'
GENRES = ['hiphop', 'rock', 'pop']

# functions

def remote_imports():
    import ast
    from pandas.api.types import CategoricalDtype
    import ipyparallel
    import librosa
    import numpy as np
    import os
    import pandas as pd
    import pywt
    from scipy.stats import skew


def load(filepath):
    """
    This method was adapted from the FMA: A Dataset For Music Analysis.
    """
    filename = os.path.basename(filepath)

    # if 'features' in filename:
    #     return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
    #
    # if 'echonest' in filename:
    #     return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
    #
    # if 'genres' in filename:
    #     return pd.read_csv(filepath, index_col=None)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
            CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks


def stats(feature):
    return {
        'mean': np.mean(feature),
        'median': np.median(feature),
        'std': np.std(feature),
        'var': np.var(feature)
    }


def extra_stats(feature):
    return {
        'sb_energy': np.mean(np.abs(feature)),
        'skewness': skew(feature)
    }

def extract_cqt(file):
    # get sample rate of audio file and load audio file as time series
    sample_rate = librosa.core.get_samplerate(file.path)
    time_series, _ = librosa.core.load(file.path, sample_rate)

    # compute cqt and convert from amplitude to decibels unit
    cqt = librosa.cqt(time_series, sample_rate)
    scaled_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    
    return scaled_cqt

def extract_mel_spect(file):
    # get sample rate of audio file and load audio file as time series
    sample_rate = librosa.core.get_samplerate(file.path)
    time_series, _ = librosa.core.load(file.path, sample_rate)

    # compute spectogram and convert spectogram to decibels unit 
    mel_spect = librosa.feature.melspectrogram(time_series, sample_rate)
    scaled_mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    
    return scaled_mel_spect

# store for all features to be extracted except log-mel and mel-spectogram.
dataframe = pd.DataFrame({
    'genre_label': [],
    'data_source': [],

    'mean_spec_centroid': [],
    'median_spec_centroid': [],
    'std_spec_centroid': [],
    'var_spec_centroid': [],

    'mean_spec_rolloff': [],
    'median_spec_rolloff': [],
    'std_spec_rolloff': [],
    'var_spec_rolloff': [],

    'mean_zcr': [],
    'median_zcr': [],
    'std_zcr': [],
    'var_zcr': [],

    'mean_spec_bw': [],
    'median_spec_bw': [],
    'std_spec_bw': [],
    'var_spec_bw': [],

    'mean_spec_contrast_1': [],
    'median_spec_contrast_1': [],
    'std_spec_contrast_1': [],
    'var_spec_contrast_1': [],

    'mean_spec_contrast_2': [],
    'median_spec_contrast_2': [],
    'std_spec_contrast_2': [],
    'var_spec_contrast_2': [],

    'mean_spec_contrast_3': [],
    'median_spec_contrast_3': [],
    'std_spec_contrast_3': [],
    'var_spec_contrast_3': [],

    'mean_spec_contrast_4': [],
    'median_spec_contrast_4': [],
    'std_spec_contrast_4': [],
    'var_spec_contrast_4': [],

    'mean_spec_contrast_5': [],
    'median_spec_contrast_5': [],
    'std_spec_contrast_5': [],
    'var_spec_contrast_5': [],

    'mean_spec_contrast_6': [],
    'median_spec_contrast_6': [],
    'std_spec_contrast_6': [],
    'var_spec_contrast_6': [],

    'mean_spec_contrast_7': [],
    'median_spec_contrast_7': [],
    'std_spec_contrast_7': [],
    'var_spec_contrast_7': [],

    'mean_mfcc_1': [],
    'median_mfcc_1': [],
    'std_mfcc_1': [],
    'var_mfcc_1': [],

    'mean_mfcc_2': [],
    'median_mfcc_2': [],
    'std_mfcc_2': [],
    'var_mfcc_2': [],

    'mean_mfcc_3': [],
    'median_mfcc_3': [],
    'std_mfcc_3': [],
    'var_mfcc_3': [],

    'mean_mfcc_4': [],
    'median_mfcc_4': [],
    'std_mfcc_4': [],
    'var_mfcc_4': [],

    'mean_mfcc_5': [],
    'median_mfcc_5': [],
    'std_mfcc_5': [],
    'var_mfcc_5': [],

    'mean_mfcc_6': [],
    'median_mfcc_6': [],
    'std_mfcc_6': [],
    'var_mfcc_6': [],

    'mean_mfcc_7': [],
    'median_mfcc_7': [],
    'std_mfcc_7': [],
    'var_mfcc_7': [],

    'mean_mfcc_8': [],
    'median_mfcc_8': [],
    'std_mfcc_8': [],
    'var_mfcc_8': [],

    'mean_mfcc_9': [],
    'median_mfcc_9': [],
    'std_mfcc_9': [],
    'var_mfcc_9': [],

    'mean_mfcc_10': [],
    'median_mfcc_10': [],
    'std_mfcc_10': [],
    'var_mfcc_10': [],

    'mean_mfcc_11': [],
    'median_mfcc_11': [],
    'std_mfcc_11': [],
    'var_mfcc_11': [],

    'mean_mfcc_12': [],
    'median_mfcc_12': [],
    'std_mfcc_12': [],
    'var_mfcc_12': [],

    'mean_mfcc_13': [],
    'median_mfcc_13': [],
    'std_mfcc_13': [],
    'var_mfcc_13': [],

    'lpc_1': [],
    'lpc_2': [],
    'lpc_3': [],
    'lpc_4': [],

    'tempo': [],

    'mean_beats': [],
    'median_beats': [],
    'std_beats': [],
    'var_beats': [],

    'mean_beats_timestamp': [],
    'median_beats_timestamp': [],
    'std_beats_timestamp': [],
    'var_beats_timestamp': [],

    'mean_db4_cA4': [],
    'median_db4_cA4': [],
    'std_db4_cA4': [],
    'var_db4_cA4': [],
    'sb_energy_db4_cA4': [],
    'skewness_db4_cA4': [],

    'mean_db4_cD4': [],
    'median_db4_cD4': [],
    'std_db4_cD4': [],
    'var_db4_cD4': [],
    'sb_energy_db4_cD4': [],
    'skewness_db4_cD4': [],

    'mean_db4_cD3': [],
    'median_db4_cD3': [],
    'std_db4_cD3': [],
    'var_db4_cD3': [],
    'sb_energy_db4_cD3': [],
    'skewness_db4_cD3': [],

    'mean_db4_cD2': [],
    'median_db4_cD2': [],
    'std_db4_cD2': [],
    'var_db4_cD2': [],
    'sb_energy_db4_cD2': [],
    'skewness_db4_cD2': [],

    'mean_db4_cD1': [],
    'median_db4_cD1': [],
    'std_db4_cD1': [],
    'var_db4_cD1': [],
    'sb_energy_db4_cD1': [],
    'skewness_db4_cD1': [],

    'mean_db5_cA4': [],
    'median_db5_cA4': [],
    'std_db5_cA4': [],
    'var_db5_cA4': [],
    'sb_energy_db5_cA4': [],
    'skewness_db5_cA4': [],

    'mean_db5_cD4': [],
    'median_db5_cD4': [],
    'std_db5_cD4': [],
    'var_db5_cD4': [],
    'sb_energy_db5_cD4': [],
    'skewness_db5_cD4': [],

    'mean_db5_cD3': [],
    'median_db5_cD3': [],
    'std_db5_cD3': [],
    'var_db5_cD3': [],
    'sb_energy_db5_cD3': [],
    'skewness_db5_cD3': [],

    'mean_db5_cD2': [],
    'median_db5_cD2': [],
    'std_db5_cD2': [],
    'var_db5_cD2': [],
    'sb_energy_db5_cD2': [],
    'skewness_db5_cD2': [],

    'mean_db5_cD1': [],
    'median_db5_cD1': [],
    'std_db5_cD1': [],
    'var_db5_cD1': [],
    'sb_energy_db5_cD1': [],
    'skewness_db5_cD1': [],

    'mean_db8_cA7': [],
    'median_db8_cA7': [],
    'std_db8_cA7': [],
    'var_db8_cA7': [],
    'sb_energy_db8_cA7': [],
    'skewness_db8_cA7': [],

    'mean_db8_cD7': [],
    'median_db8_cD7': [],
    'std_db8_cD7': [],
    'var_db8_cD7': [],
    'sb_energy_db8_cD7': [],
    'skewness_db8_cD7': [],

    'mean_db8_cD6': [],
    'median_db8_cD6': [],
    'std_db8_cD6': [],
    'var_db8_cD6': [],
    'sb_energy_db8_cD6': [],
    'skewness_db8_cD6': [],

    'mean_db8_cD5': [],
    'median_db8_cD5': [],
    'std_db8_cD5': [],
    'var_db8_cD5': [],
    'sb_energy_db8_cD5': [],
    'skewness_db8_cD5': [],

    'mean_db8_cD4': [],
    'median_db8_cD4': [],
    'std_db8_cD4': [],
    'var_db8_cD4': [],
    'sb_energy_db8_cD4': [],
    'skewness_db8_cD4': [],

    'mean_db8_cD3': [],
    'median_db8_cD3': [],
    'std_db8_cD3': [],
    'var_db8_cD3': [],
    'sb_energy_db8_cD3': [],
    'skewness_db8_cD3': [],

    'mean_db8_cD2': [],
    'median_db8_cD2': [],
    'std_db8_cD2': [],
    'var_db8_cD2': [],
    'sb_energy_db8_cD2': [],
    'skewness_db8_cD2': [],

    'mean_db8_cD1': [],
    'median_db8_cD1': [],
    'std_db8_cD1': [],
    'var_db8_cD1': [],
    'sb_energy_db8_cD1': [],
    'skewness_db8_cD1': [],

})


def extract_audio_features(dataframe, file, genre_label, data_source):
    '''
    This function takes a datafame, an audio file (check librosa for acceptable formats),
    genre label, and datasource. It extract features from the audio and returns the dataframe with
    the new row appended.

    Timbral, rhythmic, and wavelet features are extracted excluding log-mel and mel-spectogram.

    Parameters:
    dataframe (pd.Dataframe): Dataframe to update with new row.
    file (File or str): an audio file or file path.
    genre_label (str): audio genre label
    data_source (str): fma or gtzan
    '''
    
    # get sample rate of audio file & load audio file as time series
    if type(file) == str:   
        sample_rate = librosa.core.get_samplerate(file)
        time_series, _ = librosa.core.load(file, sample_rate)
    else:
        sample_rate = librosa.core.get_samplerate(file.path)
        time_series, _ = librosa.core.load(file.path, sample_rate)

    # compute timbral features
    # compute spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(time_series, sample_rate)
    stats_spec_centroid = stats(spec_centroid)

    # compute spectral roll-off
    spec_rolloff = librosa.feature.spectral_rolloff(time_series, sample_rate)
    stats_spec_rolloff = stats(spec_rolloff)

    # compute zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(time_series)
    stats_zcr = stats(zcr)

    # compute spectral bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(time_series, sample_rate)
    stats_spec_bw = stats(spec_bw[0])

    # compute spectral contrast
    spec_contrast = librosa.feature.spectral_contrast(time_series, sample_rate)
    stats_spec_contrast_1 = stats(spec_contrast[0])
    stats_spec_contrast_2 = stats(spec_contrast[1])
    stats_spec_contrast_3 = stats(spec_contrast[2])
    stats_spec_contrast_4 = stats(spec_contrast[3])
    stats_spec_contrast_5 = stats(spec_contrast[4])
    stats_spec_contrast_6 = stats(spec_contrast[5])
    stats_spec_contrast_7 = stats(spec_contrast[6])

    # compute 13 mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(time_series, sample_rate, n_mfcc=13)
    stat_mfcc_1 = stats(mfcc[0])
    stat_mfcc_2 = stats(mfcc[1])
    stat_mfcc_3 = stats(mfcc[2])
    stat_mfcc_4 = stats(mfcc[3])
    stat_mfcc_5 = stats(mfcc[4])
    stat_mfcc_6 = stats(mfcc[5])
    stat_mfcc_7 = stats(mfcc[6])
    stat_mfcc_8 = stats(mfcc[7])
    stat_mfcc_9 = stats(mfcc[8])
    stat_mfcc_10 = stats(mfcc[9])
    stat_mfcc_11 = stats(mfcc[10])
    stat_mfcc_12 = stats(mfcc[11])
    stat_mfcc_13 = stats(mfcc[12])

    # compute 3rd order linear prediction coefficients
    lpc = librosa.lpc(time_series, 3)

    # compute rhythmic features
    # compute tempo & beats
    tempo, beats = librosa.beat.beat_track(time_series, sample_rate)
    stats_beats = stats(beats)

    # compute timestamps from beats
    beats_timestamp = librosa.frames_to_time(beats, sample_rate)
    stats_beats_timestamp = stats(beats_timestamp)

    # compute wavelet features
    # compute coefficients for Db4 at level 4 decomposition
    db4_coeffs = pywt.wavedec(time_series, 'db4', level=4)
    db4_cA4, db4_cD4, db4_cD3, db4_cD2, db4_cD1 = db4_coeffs
    stats_db4_cA4 = {**stats(db4_cA4), **extra_stats(db4_cA4)}
    stats_db4_cD4 = {**stats(db4_cD4), **extra_stats(db4_cD4)}
    stats_db4_cD3 = {**stats(db4_cD3), **extra_stats(db4_cD3)}
    stats_db4_cD2 = {**stats(db4_cD2), **extra_stats(db4_cD2)}
    stats_db4_cD1 = {**stats(db4_cD1), **extra_stats(db4_cD1)}

    # compute coefficients for Db5 at level 4 decomposition
    db5_coeffs = pywt.wavedec(time_series, 'db5', level=4)
    db5_cA4, db5_cD4, db5_cD3, db5_cD2, db5_cD1 = db5_coeffs
    stats_db5_cA4 = {**stats(db5_cA4), **extra_stats(db5_cA4)}
    stats_db5_cD4 = {**stats(db5_cD4), **extra_stats(db5_cD4)}
    stats_db5_cD3 = {**stats(db5_cD3), **extra_stats(db5_cD3)}
    stats_db5_cD2 = {**stats(db5_cD2), **extra_stats(db5_cD2)}
    stats_db5_cD1 = {**stats(db5_cD1), **extra_stats(db5_cD1)}

    # compute coefficients for Db8 at level 7 decomposition
    db8_coeffs = pywt.wavedec(time_series, 'db4', level=7)
    db8_cA7, db8_cD7, db8_cD6, db8_cD5, db8_cD4, db8_cD3, db8_cD2, db8_cD1 = db8_coeffs
    stats_db8_cA7 = {**stats(db8_cA7), **extra_stats(db8_cA7)}
    stats_db8_cD7 = {**stats(db8_cD7), **extra_stats(db8_cD7)}
    stats_db8_cD6 = {**stats(db8_cD6), **extra_stats(db8_cD6)}
    stats_db8_cD5 = {**stats(db8_cD5), **extra_stats(db8_cD5)}
    stats_db8_cD4 = {**stats(db8_cD4), **extra_stats(db8_cD4)}
    stats_db8_cD3 = {**stats(db8_cD3), **extra_stats(db8_cD3)}
    stats_db8_cD2 = {**stats(db8_cD2), **extra_stats(db8_cD2)}
    stats_db8_cD1 = {**stats(db8_cD1), **extra_stats(db8_cD1)}

    # create new row
    new_row = {
        'genre_label': genre_label,
        'data_source': data_source,

        'mean_spec_centroid': stats_spec_centroid['mean'],
        'median_spec_centroid': stats_spec_centroid['median'],
        'std_spec_centroid': stats_spec_centroid['std'],
        'var_spec_centroid': stats_spec_centroid['var'],

        'mean_spec_rolloff': stats_spec_rolloff['mean'],
        'median_spec_rolloff': stats_spec_rolloff['median'],
        'std_spec_rolloff': stats_spec_rolloff['std'],
        'var_spec_rolloff': stats_spec_rolloff['var'],

        'mean_zcr': stats_zcr['mean'],
        'median_zcr': stats_zcr['median'],
        'std_zcr': stats_zcr['std'],
        'var_zcr': stats_zcr['var'],

        'mean_spec_bw': stats_spec_bw['mean'],
        'median_spec_bw': stats_spec_bw['median'],
        'std_spec_bw': stats_spec_bw['std'],
        'var_spec_bw': stats_spec_bw['var'],

        'mean_spec_contrast_1': stats_spec_contrast_1['mean'],
        'median_spec_contrast_1': stats_spec_contrast_1['median'],
        'std_spec_contrast_1': stats_spec_contrast_1['std'],
        'var_spec_contrast_1': stats_spec_contrast_1['var'],

        'mean_spec_contrast_2': stats_spec_contrast_2['mean'],
        'median_spec_contrast_2': stats_spec_contrast_2['median'],
        'std_spec_contrast_2': stats_spec_contrast_2['std'],
        'var_spec_contrast_2': stats_spec_contrast_2['var'],

        'mean_spec_contrast_3': stats_spec_contrast_3['mean'],
        'median_spec_contrast_3': stats_spec_contrast_3['median'],
        'std_spec_contrast_3': stats_spec_contrast_3['std'],
        'var_spec_contrast_3': stats_spec_contrast_3['var'],

        'mean_spec_contrast_4': stats_spec_contrast_4['mean'],
        'median_spec_contrast_4': stats_spec_contrast_4['median'],
        'std_spec_contrast_4': stats_spec_contrast_4['std'],
        'var_spec_contrast_4': stats_spec_contrast_4['var'],

        'mean_spec_contrast_5': stats_spec_contrast_5['mean'],
        'median_spec_contrast_5': stats_spec_contrast_5['median'],
        'std_spec_contrast_5': stats_spec_contrast_5['std'],
        'var_spec_contrast_5': stats_spec_contrast_5['var'],

        'mean_spec_contrast_6': stats_spec_contrast_6['mean'],
        'median_spec_contrast_6': stats_spec_contrast_6['median'],
        'std_spec_contrast_6': stats_spec_contrast_6['std'],
        'var_spec_contrast_6': stats_spec_contrast_6['var'],

        'mean_spec_contrast_7': stats_spec_contrast_7['mean'],
        'median_spec_contrast_7': stats_spec_contrast_7['median'],
        'std_spec_contrast_7': stats_spec_contrast_7['std'],
        'var_spec_contrast_7': stats_spec_contrast_7['var'],

        'mean_mfcc_1': stat_mfcc_1['mean'],
        'median_mfcc_1': stat_mfcc_1['median'],
        'std_mfcc_1': stat_mfcc_1['std'],
        'var_mfcc_1': stat_mfcc_1['var'],

        'mean_mfcc_2': stat_mfcc_2['mean'],
        'median_mfcc_2': stat_mfcc_2['median'],
        'std_mfcc_2': stat_mfcc_2['std'],
        'var_mfcc_2': stat_mfcc_2['var'],

        'mean_mfcc_3': stat_mfcc_3['mean'],
        'median_mfcc_3': stat_mfcc_3['median'],
        'std_mfcc_3': stat_mfcc_3['std'],
        'var_mfcc_3': stat_mfcc_3['var'],

        'mean_mfcc_4': stat_mfcc_4['mean'],
        'median_mfcc_4': stat_mfcc_4['median'],
        'std_mfcc_4': stat_mfcc_4['std'],
        'var_mfcc_4': stat_mfcc_4['var'],

        'mean_mfcc_5': stat_mfcc_5['mean'],
        'median_mfcc_5': stat_mfcc_5['median'],
        'std_mfcc_5': stat_mfcc_5['std'],
        'var_mfcc_5': stat_mfcc_5['var'],

        'mean_mfcc_6': stat_mfcc_6['mean'],
        'median_mfcc_6': stat_mfcc_6['median'],
        'std_mfcc_6': stat_mfcc_6['std'],
        'var_mfcc_6': stat_mfcc_6['var'],

        'mean_mfcc_7': stat_mfcc_7['mean'],
        'median_mfcc_7': stat_mfcc_7['median'],
        'std_mfcc_7': stat_mfcc_7['std'],
        'var_mfcc_7': stat_mfcc_7['var'],

        'mean_mfcc_8': stat_mfcc_8['mean'],
        'median_mfcc_8': stat_mfcc_8['median'],
        'std_mfcc_8': stat_mfcc_8['std'],
        'var_mfcc_8': stat_mfcc_8['var'],

        'mean_mfcc_9': stat_mfcc_9['mean'],
        'median_mfcc_9': stat_mfcc_9['median'],
        'std_mfcc_9': stat_mfcc_9['std'],
        'var_mfcc_9': stat_mfcc_9['var'],

        'mean_mfcc_10': stat_mfcc_10['mean'],
        'median_mfcc_10': stat_mfcc_10['median'],
        'std_mfcc_10': stat_mfcc_10['std'],
        'var_mfcc_10': stat_mfcc_10['var'],

        'mean_mfcc_11': stat_mfcc_11['mean'],
        'median_mfcc_11': stat_mfcc_11['median'],
        'std_mfcc_11': stat_mfcc_11['std'],
        'var_mfcc_11': stat_mfcc_11['var'],

        'mean_mfcc_12': stat_mfcc_12['mean'],
        'median_mfcc_12': stat_mfcc_12['median'],
        'std_mfcc_12': stat_mfcc_12['std'],
        'var_mfcc_12': stat_mfcc_12['var'],

        'mean_mfcc_13': stat_mfcc_13['mean'],
        'median_mfcc_13': stat_mfcc_13['median'],
        'std_mfcc_13': stat_mfcc_13['std'],
        'var_mfcc_13': stat_mfcc_13['var'],

        'lpc_1': lpc[0],
        'lpc_2': lpc[1],
        'lpc_3': lpc[2],
        'lpc_4': lpc[3],

        'tempo': tempo,

        'mean_beats': stats_beats['mean'],
        'median_beats': stats_beats['median'],
        'std_beats': stats_beats['std'],
        'var_beats': stats_beats['var'],

        'mean_beats_timestamp': stats_beats_timestamp['mean'],
        'median_beats_timestamp': stats_beats_timestamp['median'],
        'std_beats_timestamp': stats_beats_timestamp['std'],
        'var_beats_timestamp': stats_beats_timestamp['var'],

        'mean_db4_cA4': stats_db4_cA4['mean'],
        'median_db4_cA4': stats_db4_cA4['median'],
        'std_db4_cA4': stats_db4_cA4['std'],
        'var_db4_cA4': stats_db4_cA4['var'],
        'sb_energy_db4_cA4': stats_db4_cA4['sb_energy'],
        'skewness_db4_cA4': stats_db4_cA4['skewness'],

        'mean_db4_cD4': stats_db4_cD4['mean'],
        'median_db4_cD4': stats_db4_cD4['median'],
        'std_db4_cD4': stats_db4_cD4['std'],
        'var_db4_cD4': stats_db4_cD4['var'],
        'sb_energy_db4_cD4': stats_db4_cD4['sb_energy'],
        'skewness_db4_cD4': stats_db4_cD4['skewness'],

        'mean_db4_cD3': stats_db4_cD3['mean'],
        'median_db4_cD3': stats_db4_cD3['median'],
        'std_db4_cD3': stats_db4_cD3['std'],
        'var_db4_cD3': stats_db4_cD3['var'],
        'sb_energy_db4_cD3': stats_db4_cD3['sb_energy'],
        'skewness_db4_cD3': stats_db4_cD3['skewness'],

        'mean_db4_cD2': stats_db4_cD2['mean'],
        'median_db4_cD2': stats_db4_cD2['median'],
        'std_db4_cD2': stats_db4_cD2['std'],
        'var_db4_cD2': stats_db4_cD2['var'],
        'sb_energy_db4_cD2': stats_db4_cD2['sb_energy'],
        'skewness_db4_cD2': stats_db4_cD2['skewness'],

        'mean_db4_cD1': stats_db4_cD1['mean'],
        'median_db4_cD1': stats_db4_cD1['median'],
        'std_db4_cD1': stats_db4_cD1['std'],
        'var_db4_cD1': stats_db4_cD1['var'],
        'sb_energy_db4_cD1': stats_db4_cD1['sb_energy'],
        'skewness_db4_cD1': stats_db4_cD1['skewness'],

        'mean_db5_cA4': stats_db5_cA4['mean'],
        'median_db5_cA4': stats_db5_cA4['median'],
        'std_db5_cA4': stats_db5_cA4['std'],
        'var_db5_cA4': stats_db5_cA4['var'],
        'sb_energy_db5_cA4': stats_db5_cA4['sb_energy'],
        'skewness_db5_cA4': stats_db5_cA4['skewness'],

        'mean_db5_cD4': stats_db5_cD4['mean'],
        'median_db5_cD4': stats_db5_cD4['median'],
        'std_db5_cD4': stats_db5_cD4['std'],
        'var_db5_cD4': stats_db5_cD4['var'],
        'sb_energy_db5_cD4': stats_db5_cD4['sb_energy'],
        'skewness_db5_cD4': stats_db5_cD4['skewness'],

        'mean_db5_cD3': stats_db5_cD3['mean'],
        'median_db5_cD3': stats_db5_cD3['median'],
        'std_db5_cD3': stats_db5_cD3['std'],
        'var_db5_cD3': stats_db5_cD3['var'],
        'sb_energy_db5_cD3': stats_db5_cD3['sb_energy'],
        'skewness_db5_cD3': stats_db5_cD3['skewness'],

        'mean_db5_cD2': stats_db5_cD2['mean'],
        'median_db5_cD2': stats_db5_cD2['median'],
        'std_db5_cD2': stats_db5_cD2['std'],
        'var_db5_cD2': stats_db5_cD2['var'],
        'sb_energy_db5_cD2': stats_db5_cD2['sb_energy'],
        'skewness_db5_cD2': stats_db5_cD2['skewness'],

        'mean_db5_cD1': stats_db5_cD1['mean'],
        'median_db5_cD1': stats_db5_cD1['median'],
        'std_db5_cD1': stats_db5_cD1['std'],
        'var_db5_cD1': stats_db5_cD1['var'],
        'sb_energy_db5_cD1': stats_db5_cD1['sb_energy'],
        'skewness_db5_cD1': stats_db5_cD1['skewness'],

        'mean_db8_cA7': stats_db8_cA7['mean'],
        'median_db8_cA7': stats_db8_cA7['median'],
        'std_db8_cA7': stats_db8_cA7['std'],
        'var_db8_cA7': stats_db8_cA7['var'],
        'sb_energy_db8_cA7': stats_db8_cA7['sb_energy'],
        'skewness_db8_cA7': stats_db8_cA7['skewness'],

        'mean_db8_cD7': stats_db8_cD7['mean'],
        'median_db8_cD7': stats_db8_cD7['median'],
        'std_db8_cD7': stats_db8_cD7['std'],
        'var_db8_cD7': stats_db8_cD7['var'],
        'sb_energy_db8_cD7': stats_db8_cD7['sb_energy'],
        'skewness_db8_cD7': stats_db8_cD7['skewness'],

        'mean_db8_cD6': stats_db8_cD6['mean'],
        'median_db8_cD6': stats_db8_cD6['median'],
        'std_db8_cD6': stats_db8_cD6['std'],
        'var_db8_cD6': stats_db8_cD6['var'],
        'sb_energy_db8_cD6': stats_db8_cD6['sb_energy'],
        'skewness_db8_cD6': stats_db8_cD6['skewness'],

        'mean_db8_cD5': stats_db8_cD5['mean'],
        'median_db8_cD5': stats_db8_cD5['median'],
        'std_db8_cD5': stats_db8_cD5['std'],
        'var_db8_cD5': stats_db8_cD5['var'],
        'sb_energy_db8_cD5': stats_db8_cD5['sb_energy'],
        'skewness_db8_cD5': stats_db8_cD5['skewness'],

        'mean_db8_cD4': stats_db8_cD4['mean'],
        'median_db8_cD4': stats_db8_cD4['median'],
        'std_db8_cD4': stats_db8_cD4['std'],
        'var_db8_cD4': stats_db8_cD4['var'],
        'sb_energy_db8_cD4': stats_db8_cD4['sb_energy'],
        'skewness_db8_cD4': stats_db8_cD4['skewness'],

        'mean_db8_cD3': stats_db8_cD3['mean'],
        'median_db8_cD3': stats_db8_cD3['median'],
        'std_db8_cD3': stats_db8_cD3['std'],
        'var_db8_cD3': stats_db8_cD3['var'],
        'sb_energy_db8_cD3': stats_db8_cD3['sb_energy'],
        'skewness_db8_cD3': stats_db8_cD3['skewness'],

        'mean_db8_cD2': stats_db8_cD2['mean'],
        'median_db8_cD2': stats_db8_cD2['median'],
        'std_db8_cD2': stats_db8_cD2['std'],
        'var_db8_cD2': stats_db8_cD2['var'],
        'sb_energy_db8_cD2': stats_db8_cD2['sb_energy'],
        'skewness_db8_cD2': stats_db8_cD2['skewness'],

        'mean_db8_cD1': stats_db8_cD1['mean'],
        'median_db8_cD1': stats_db8_cD1['median'],
        'std_db8_cD1': stats_db8_cD1['std'],
        'var_db8_cD1': stats_db8_cD1['var'],
        'sb_energy_db8_cD1': stats_db8_cD1['sb_energy'],
        'skewness_db8_cD1': stats_db8_cD1['skewness'],
    }

    # append new row
    dataframe = dataframe.append(new_row, ignore_index=True)
    
    if type(file) == str:   
        print('appended features extracted from ' + file + ' with genre: ' + genre_label)
    else:
        print('appended features extracted from ' + str(file.name) + ' with genre: ' + genre_label)

    return dataframe
