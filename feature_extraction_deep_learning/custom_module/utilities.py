import os

import ast
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
from scipy.stats import entropy, skew
from sklearn.decomposition import PCA
import joblib
from tensorflow.keras import layers, optimizers, callbacks
import librosa
from tensorflow.keras.models import Sequential, load_model
from sklearn.metrics import ( 
    confusion_matrix, accuracy_score, 
    precision_recall_fscore_support,
    roc_curve, auc
)
from sklearn.model_selection import RandomizedSearchCV
import numpy
import pandas
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import matplotlib.pyplot as pyplot
import pywt
import seaborn
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from pandas.api.types import CategoricalDtype


# constants
SYS_DIR_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
MOUNTED_DATASET_PATH = SYS_DIR_PATH + '/s3-bucket'
SAMPLE_HIPHOP_FILE_PATH = MOUNTED_DATASET_PATH + '/gtzan/wavfiles/hiphop.00000.wav'
SAMPLE_POP_FILE_PATH = MOUNTED_DATASET_PATH + '/gtzan/wavfiles/pop.00000.wav'
SAMPLE_ROCK_FILE_PATH = MOUNTED_DATASET_PATH + '/gtzan/wavfiles/rock.00000.wav'
GENRES = ['hiphop', 'rock', 'pop']
GENRES_MAP = {
    0: 'hiphop',
    1: 'rock',
    2: 'pop'
}
# set log directory for tensorboard logs
root_logdir = os.path.join(os.curdir, "my_logs")


# utility classes and functions
def visualize_conf_metrics(X_test, y_test, model):
    '''
    visualize_conf_metrics() plot the confusion metrics
    taking as parameter the test data.
    '''
    # make prediction on test data
    predicted_y_test = model.predict_proba(X_test)

    # compute confusion matrix
    conf_matrix = confusion_matrix(
        numpy.argmax(y_test, 1), numpy.argmax(predicted_y_test, 1))
    conf_matrix = pandas.DataFrame(conf_matrix)
    conf_matrix = conf_matrix.rename(columns=GENRES_MAP)
    conf_matrix.index = conf_matrix.columns
    
    # plot confusion matrix
    pyplot.figure(figsize= (20,12))
    seaborn.set(font_scale = 2);
    ax = seaborn.heatmap(conf_matrix, annot=True, cmap=seaborn.cubehelix_palette(50));
    ax.set(xlabel='Predicted Values', ylabel='Actual Values');
    
    
def evaluate_model(X_test, y_test, model):
    """
    evaluate_model() returns evaluation metrics to 
    measure the performace of the model. Taking as 
    parameter test data and model instance.
    """
    mean_fpr = numpy.linspace(start=0, stop=1, num=100)
    
    # compute probabilistic predictiond for the evaluation set
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # compute exact predictiond for the evaluation set
    predicted_values = model.predict(X_test)
        
    # compute accuracy
    accuracy = accuracy_score(y_test, predicted_values)
        
    # compute precision, recall and f1 score for class 1
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predicted_values, labels=[1])
    
#     # compute fpr and tpr values for various thresholds 
#     # by comparing the true target values to the predicted probabilities for class 1
#     fpr, tpr, _ = roc_curve(y_test, probabilities)
        
#     # compute true positive rates for the values in the array mean_fpr
#     tpr_transformed = np.array([interp(mean_fpr, fpr, tpr)])
    
#     # compute the area under the curve
#     auc = auc(fpr, tpr)
            
    return accuracy, precision[0], recall[0], f1_score[0]


def get_run_logdir(): 
    '''
    get_run_logdir() generates subdirectory path with
    current date & time.
    '''   
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S") 
    return os.path.join(root_logdir, run_id)


def training_best_model(X_train, y_train, model_name, ncols,
                        build_fn, preprocess_pipeline, params, batch_size=32,
                        epochs=100):
    """
    training_best_model() trains the best model
    from each subsets and returns an instance
    of the model. Taking as params things you 
    need to train a model.
    """
    # create early stopping callback instance
    early_stopping_cb = callbacks.EarlyStopping(
        patience=10, restore_best_weights=True)
    
    # generate log dir and create tensorboard callback instance 
    run_logdir = get_run_logdir() # e.g., './my_logs/run_2019_01_16-11_28_43'
    tensorboard_cb = callbacks.TensorBoard(run_logdir) 

    # wrap the function with keras wrapper
    clf = KerasClassifier(build_fn=build_fn(model_name, ncols))
    
    # pass-in params to be used to create model
    clf.set_params(**params)

    # create pipeline estimator
    estimator = Pipeline([
        ('preprocess', preprocess_pipeline),
        ('clf', clf)
    ])

    # learn model & validate with 20% of the training dataset
    estimator.fit(X_train, y_train, clf__batch_size=batch_size,
                  clf__validation_split=0.2, clf__epochs=epochs,
                  clf__callbacks=[early_stopping_cb, tensorboard_cb])

    return estimator


def create_norm_pipelines(features):
    """
    create_norm_pipelines() returns a dict of 
    normalization pipeline instances of four 
    subsets of the dataset taking as parameter
    features (pandas.DataFrame).
    """
    # break into subset
    (timbral_rhythmic_predictors, predictors_with_pos_corr,
     wavelet_predictors) = break_into_subsets(features)
    all_predictors = features.drop('genre_label', axis=1)
    
    # create dict of pipeline instances for each subset     
    norm_pipelines = {
        'all': normalization_pipeline(
            all_predictors.columns.values),
        'tr': normalization_pipeline(
            timbral_rhythmic_predictors.columns.values),
        'pos_corr': normalization_pipeline(
            predictors_with_pos_corr.columns.values),
        'wavelet': normalization_pipeline(
            wavelet_predictors.columns.values)
    }
    
    return norm_pipelines


def break_into_subsets(features):
    """
    break_into_subsets() returns subsets of the dataset
    taking as parameter features (pandas.DataFrame).
    """
    
    # get wavelet subset     
    wavelet_predictors = features.filter(regex=(r'.+_db[458]{1}_.+'))
    
    # get timbral & rhytmic subset
    wavelet_predictors_labels = wavelet_predictors.columns.values
    timbral_rhythmic_predictors = features.loc[:, features.columns.difference(
        numpy.append(wavelet_predictors_labels, 'genre_label'))]
    
    # get features with +ve correlation with the target subset
    corr_wf_target = features.corr()[['genre_label']].sort_values(
        by=['genre_label'], ascending=False)
    predictor_labels_with_pos_corr = corr_wf_target.loc[
        corr_wf_target.loc[:,'genre_label'] > 0].index.values
    predictors_wf_pos_corr = features.loc[
        :, predictor_labels_with_pos_corr].drop('genre_label', axis=1)
    
    return timbral_rhythmic_predictors, predictors_wf_pos_corr, wavelet_predictors


def train_model(X, y, model_name, ncols, build_fn, preprocess_pipeline,
                param_dist, batch_size=32, epochs=100):
    """
    train_model() trains a model using cv and return
    the best model score, params, & instance. Takes as
    parameter model name (str), batch_size (int), epochs (int).
    """
    # create early stopping callback instance
    early_stopping_cb = callbacks.EarlyStopping(
        patience=10, restore_best_weights=True)

    # wrap the function with keras wrapper
    clf = KerasClassifier(build_fn=build_fn(model_name, ncols))

    # create pipeline estimator
    pipeline = Pipeline([
        ('preprocess', preprocess_pipeline),
        ('clf', clf)
    ])

    # instantiate RandomizedSearchCV
    # if you're not using a GPU, you can set n_jobs to something other than 1
    rscv = RandomizedSearchCV(pipeline, param_dist, cv=3, n_jobs=1)

    # learn model & validate with 20% of the training dataset
    search = rscv.fit(X, y, clf__batch_size=batch_size,
                      clf__validation_split=0.2, clf__epochs=epochs,
                      clf__callbacks=[early_stopping_cb])

    print(search.best_score_, search.best_params_)

    return search.best_score_, search.best_params_, search.best_estimator_


def visualize_wavelet(signal, waveletname, level_of_dec):
    """
    visualize_wavelet() create original visualization for a signal and 
    create visulization of the signal for a specified wavelet type for 
    the range of decompsition level specified taking as parameter the
    signal, wavelet name and max level of decomposition. 
    
    It was adapted from Ahmet Taspinar's blog titled:
    'A guide for using the Wavelet Transform in Machine Learning'.
    """
    # visualize orignal signal
    fig, ax = pyplot.subplots(figsize=(10, 5))
    ax.set_title("Music Signal: ")
    ax.plot(signal)

    # visualize specified wavelet for 1 - specified levels of decomposition
    cA = signal
    fig, axarr = pyplot.subplots(nrows=level_of_dec, ncols=2, figsize=(10, level_of_dec * 1.5))
    for level in range(level_of_dec):
        (cA, cD) = pywt.dwt(cA, waveletname)
        axarr[level, 0].plot(cA, 'r')
        axarr[level, 1].plot(cD, 'g')
        axarr[level, 0].set_ylabel("Level {}".format(level + 1), fontsize=14, rotation=90)
        axarr[level, 0].set_yticklabels([])
        if level == 0:
            axarr[level, 0].set_title("Approximation coefficients", fontsize=14)
            axarr[level, 1].set_title("Detail coefficients", fontsize=14)
        axarr[level, 1].set_yticklabels([])
    pyplot.tight_layout()
    pyplot.show()


def get_optimizer(lr, optimizer):
    """
    get_optimizer() returns a optimizer instance taking as parameter
    learning_rate (int) and optimizer (str)
    """
    optimizer_instances = {
        'rmsprop': optimizers.RMSprop(lr=lr),
        'adam': optimizers.Adam(lr=lr),
        'adagrad': optimizers.Adagrad(lr=lr)
    }
    return optimizer_instances[optimizer]


def set_shape_create_cnn_model(name, ncols):
    """
    set_shape_create_model() returns a create_cnn_model function,
    taking as parameters the model name and column input shape.
    """

    def create_cnn_model(n_hidden=1, activation='relu', optimizer='adam',
                         kernel_initializer='glorot_uniform', units=30,
                         filters=16, kernel_size=3, dropout=0.25, lr=3):
        """
        create_cnn_model() returns a CNN model,
        taking as parameters things you want to verify
        using cross validation and model selection
        """
        # name = 'cnn'
        # ncols = 217

        # initialize a random seed for replication purposes
        numpy.random.seed(23456)
        tf.random.set_seed(123)

        model_layers = [
            layers.Conv1D(
                filters=filters, kernel_size=kernel_size,
                input_shape=[ncols, 1]),
        ]

        index = 1
        for layer in range(n_hidden):
            index = index + 1
            # add a maxpooling layer
            model_layers.append(layers.MaxPooling1D())
            # add convolution layer
            model_layers.append(
                layers.Conv1D(
                    filters=index * filters, kernel_size=kernel_size,
                    activation=activation)
            )

        model_layers.append(layers.MaxPooling1D())
        model_layers.append(layers.Flatten())

        for layer in range(n_hidden):
            model_layers.append(
                layers.Dense(units, activation=activation,
                             kernel_initializer=kernel_initializer)
            )
            model_layers.append(layers.Dropout(dropout))

        # add an output layer
        model_layers.append(layers.Dense(3, activation='softmax'))

        # Initiating an empty NN
        model = Sequential(layers=model_layers, name=name)

        print(model.summary())

        # Compiling our NN
        model.compile(loss='categorical_crossentropy',
                      optimizer=get_optimizer(lr, optimizer),
                      metrics=['accuracy'])

        return model

    return create_cnn_model


def set_shape_create_model(name, ncols):
    """
    set_shape_create_model() returns a create_model function, 
    taking as parameters the model name and column input shape.
    """

    def create_model(n_hidden=1, activation='relu', optimizer='adam',
                     kernel_initializer='glorot_uniform', lr=3, units=30):
        """
        create_model() returns a FNN model, 
        taking as parameters things you
        want to verify using cross-validation and model selection
        """
        # initialize a random seed for replication purposes
        numpy.random.seed(23456)
        tf.random.set_seed(123)

        model_layers = [
            layers.Flatten(input_shape=[ncols, 1]),
        ]

        # multiplier = n_hidden + 1
        # units = 3 * n_hidden
        for layer in range(n_hidden):
            # add a full-connected layer
            # units = 3 * multiplier * multiplier
            model_layers.append(
                layers.Dense(units, activation=activation,
                             kernel_initializer=kernel_initializer)
            )
            # multiplier -= 1

        # add an output layer
        model_layers.append(layers.Dense(3, activation='softmax'))

        # Initiating the NN
        model = Sequential(layers=model_layers, name=name)

        print(model.summary())

        # Compiling the NN
        model.compile(loss='categorical_crossentropy',
                      optimizer=get_optimizer(lr, optimizer),
                      metrics=['accuracy'])

        return model

    return create_model


class AddColumnNames(BaseEstimator, TransformerMixin):
    """
    AddColumnNames is used to add columns to a pipeline.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pandas.DataFrame(data=X, columns=self.columns)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    ColumnSelector is used to select columns in a pipeline
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pandas.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class Reshape1DTo2D(BaseEstimator, TransformerMixin):
    """
    Reshape2D reshapes 1D input to 2D.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, numpy.ndarray)
        nrows, ncols = X.shape
        return X.reshape(nrows, ncols, 1)


class LogTransformation(BaseEstimator, TransformerMixin):
    """
    Log Transformation performs log transformation on input dataframe
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pandas.core.frame.DataFrame)

        # silence the SettingWithCopyWarning as values are
        # being set the right way using .loc[] not [].
        # The interpreter doesnt know the difference so
        # it would've displayed the warning nonetheless
        pandas.options.mode.chained_assignment = None

        labels = X.columns.values
        for label in labels:
            curr_column = X.loc[:, label]
            min_val = curr_column.min()
            increment = numpy.abs(min_val) + 1
            X.loc[:, label] = numpy.log(curr_column + increment)

        return X


# def standardization_pipeline(predictors_all, predictors_with_outliers,
#                     predictors_without_outliers, n_components):
def standardization_pipeline(predictors_all, predictors_with_outliers,
                             predictors_without_outliers):
    """
    standardization_pipeline() returns a Pipeline object, 
    taking as parameters the all labels in the dataset,
    labels of predictors with outliers, and ones without
    outliers.
    """
    if len(predictors_with_outliers) == 0:
        return make_pipeline(
            AddColumnNames(columns=predictors_all),
            FeatureUnion(transformer_list=[
                ('predictors_without_outliers', make_pipeline(
                    ColumnSelector(columns=predictors_without_outliers),
                    StandardScaler()
                ))
            ]),
            #             PCA(n_components=n_components),
            Reshape1DTo2D()
        )

    return make_pipeline(
        AddColumnNames(columns=predictors_all),
        FeatureUnion(transformer_list=[
            ('predictors_with_outliers', make_pipeline(
                ColumnSelector(columns=predictors_with_outliers),
                LogTransformation(),
                RobustScaler()
            )),
            ('predictors_without_outliers', make_pipeline(
                ColumnSelector(columns=predictors_without_outliers),
                StandardScaler()
            )),
        ]),
        #         PCA(n_components=n_components),
        Reshape1DTo2D()
    )


# def normalization_pipeline(predictors_all, n_components):
def normalization_pipeline(predictors_all):
    """
    normalization_pipeline() returns a Pipeline object, 
    taking as parameters the all labels in the dataset.
    """
    return make_pipeline(
        AddColumnNames(columns=predictors_all),
        FeatureUnion(transformer_list=[
            ('predictors_all', make_pipeline(
                ColumnSelector(columns=predictors_all),
                MinMaxScaler()
            ))
        ]),
        #         PCA(n_components=n_components),
        Reshape1DTo2D()
    )


def remote_imports():
    import ast
    from pandas.api.types import CategoricalDtype
    import ipyparallel
    import librosa
    import numpy as numpy
    import os
    import pandas as pandas
    import pywt
    from scipy.stats import skew


def load(filepath):
    """
    load() was adapted from the FMA: A Dataset For Music Analysis repository.
    It is used to load the tracks.csv file from the FMA dataset.
    """
    filename = os.path.basename(filepath)
    if 'tracks' in filename:
        tracks = pandas.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pandas.to_datetime(tracks[column])

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
    """
    stats() returns a dict of stats: mean, median, std & variance,
    for a single feature.
    """
    return {
        'mean': numpy.mean(feature),
        'median': numpy.median(feature),
        'std': numpy.std(feature),
        'var': numpy.var(feature)
    }


def calc_entropy(feature):
    """
    calc_entropy() returns the entropy for a feature list.
    It was adapted from Ahmet Taspinar's blog titled:
    'A guide for using the Wavelet Transform in Machine Learning'.
    """
    counter_values = Counter(feature).most_common()
    probabilities = [elem[1] / len(feature) for elem in counter_values]
    entropy_val = entropy(probabilities)
    return entropy_val


def extra_stats(feature):
    """
    extra_stats() returns a dict of stats which include;
    sub-band energy, skewness, (5th, 25th, 75th & 95th) percentile,
    root mean square, zero crossing rate, mean crossing rate,
    and entropy, for a single feature list.
    """
    return {
        'sb_energy': numpy.mean(numpy.abs(feature)),
        'skewness': skew(feature),
        '5th_percentile': numpy.nanpercentile(feature, 5),
        '25th_percentile': numpy.nanpercentile(feature, 25),
        '75th_percentile': numpy.nanpercentile(feature, 75),
        '95th_percentile': numpy.nanpercentile(feature, 95),
        'rms': numpy.nanmean(numpy.sqrt(feature ** 2)),
        'zcr': len(numpy.nonzero(numpy.diff(numpy.array(feature) > 0))[0]),
        'mcr': len(numpy.nonzero(numpy.diff(numpy.array(feature) > numpy.nanmean(feature)))[0]),
        'entropy': calc_entropy(feature),
    }


def extract_cqt(file):
    """
    extract_cqt() returns the log-mel value for audio signal, 
    taking as parameters file object of an audio signal
    """
    # get sample rate of audio file and load audio file as time series
    sample_rate = librosa.core.get_samplerate(file.path);
    time_series, _ = librosa.core.load(file.path, sample_rate);

    # compute cqt and convert from amplitude to decibels unit
    cqt = librosa.cqt(time_series, sample_rate);
    scaled_cqt = librosa.amplitude_to_db(cqt, ref=numpy.max);

    return scaled_cqt


def extract_mel_spect(file):
    """
    extract_mel_spect() returns the mel spectogram value for an audio signal, 
    taking as parameters file object of an audio signal
    """
    # get sample rate of audio file and load audio file as time series
    sample_rate = librosa.core.get_samplerate(file.path);
    time_series, _ = librosa.core.load(file.path, sample_rate);

    # compute spectogram and convert spectogram to decibels unit 
    mel_spect = librosa.feature.melspectrogram(time_series, sample_rate);
    scaled_mel_spect = librosa.power_to_db(mel_spect, ref=numpy.max);

    return scaled_mel_spect


# store for all features to be extracted except log-mel and mel-spectrogram.
dataframe = pandas.DataFrame({
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
    '5th_percentile_db4_cA4': [],
    '25th_percentile_db4_cA4': [],
    '75th_percentile_db4_cA4': [],
    '95th_percentile_db4_cA4': [],
    'rms_db4_cA4': [],
    'zcr_db4_cA4': [],
    'mcr_db4_cA4': [],
    'entropy_db4_cA4': [],

    'mean_db4_cD4': [],
    'median_db4_cD4': [],
    'std_db4_cD4': [],
    'var_db4_cD4': [],
    'sb_energy_db4_cD4': [],
    'skewness_db4_cD4': [],
    '5th_percentile_db4_cD4': [],
    '25th_percentile_db4_cD4': [],
    '75th_percentile_db4_cD4': [],
    '95th_percentile_db4_cD4': [],
    'rms_db4_cD4': [],
    'zcr_db4_cD4': [],
    'mcr_db4_cD4': [],
    'entropy_db4_cD4': [],

    'mean_db4_cD3': [],
    'median_db4_cD3': [],
    'std_db4_cD3': [],
    'var_db4_cD3': [],
    'sb_energy_db4_cD3': [],
    'skewness_db4_cD3': [],
    '5th_percentile_db4_cD3': [],
    '25th_percentile_db4_cD3': [],
    '75th_percentile_db4_cD3': [],
    '95th_percentile_db4_cD3': [],
    'rms_db4_cD3': [],
    'zcr_db4_cD3': [],
    'mcr_db4_cD3': [],
    'entropy_db4_cD3': [],

    'mean_db4_cD2': [],
    'median_db4_cD2': [],
    'std_db4_cD2': [],
    'var_db4_cD2': [],
    'sb_energy_db4_cD2': [],
    'skewness_db4_cD2': [],
    '5th_percentile_db4_cD2': [],
    '25th_percentile_db4_cD2': [],
    '75th_percentile_db4_cD2': [],
    '95th_percentile_db4_cD2': [],
    'rms_db4_cD2': [],
    'zcr_db4_cD2': [],
    'mcr_db4_cD2': [],
    'entropy_db4_cD2': [],

    'mean_db4_cD1': [],
    'median_db4_cD1': [],
    'std_db4_cD1': [],
    'var_db4_cD1': [],
    'sb_energy_db4_cD1': [],
    'skewness_db4_cD1': [],
    '5th_percentile_db4_cD1': [],
    '25th_percentile_db4_cD1': [],
    '75th_percentile_db4_cD1': [],
    '95th_percentile_db4_cD1': [],
    'rms_db4_cD1': [],
    'zcr_db4_cD1': [],
    'mcr_db4_cD1': [],
    'entropy_db4_cD1': [],

    'mean_db5_cA4': [],
    'median_db5_cA4': [],
    'std_db5_cA4': [],
    'var_db5_cA4': [],
    'sb_energy_db5_cA4': [],
    'skewness_db5_cA4': [],
    '5th_percentile_db5_cA4': [],
    '25th_percentile_db5_cA4': [],
    '75th_percentile_db5_cA4': [],
    '95th_percentile_db5_cA4': [],
    'rms_db5_cA4': [],
    'zcr_db5_cA4': [],
    'mcr_db5_cA4': [],
    'entropy_db5_cA4': [],

    'mean_db5_cD4': [],
    'median_db5_cD4': [],
    'std_db5_cD4': [],
    'var_db5_cD4': [],
    'sb_energy_db5_cD4': [],
    'skewness_db5_cD4': [],
    '5th_percentile_db5_cD4': [],
    '25th_percentile_db5_cD4': [],
    '75th_percentile_db5_cD4': [],
    '95th_percentile_db5_cD4': [],
    'rms_db5_cD4': [],
    'zcr_db5_cD4': [],
    'mcr_db5_cD4': [],
    'entropy_db5_cD4': [],

    'mean_db5_cD3': [],
    'median_db5_cD3': [],
    'std_db5_cD3': [],
    'var_db5_cD3': [],
    'sb_energy_db5_cD3': [],
    'skewness_db5_cD3': [],
    '5th_percentile_db5_cD3': [],
    '25th_percentile_db5_cD3': [],
    '75th_percentile_db5_cD3': [],
    '95th_percentile_db5_cD3': [],
    'rms_db5_cD3': [],
    'zcr_db5_cD3': [],
    'mcr_db5_cD3': [],
    'entropy_db5_cD3': [],

    'mean_db5_cD2': [],
    'median_db5_cD2': [],
    'std_db5_cD2': [],
    'var_db5_cD2': [],
    'sb_energy_db5_cD2': [],
    'skewness_db5_cD2': [],
    '5th_percentile_db5_cD2': [],
    '25th_percentile_db5_cD2': [],
    '75th_percentile_db5_cD2': [],
    '95th_percentile_db5_cD2': [],
    'rms_db5_cD2': [],
    'zcr_db5_cD2': [],
    'mcr_db5_cD2': [],
    'entropy_db5_cD2': [],

    'mean_db5_cD1': [],
    'median_db5_cD1': [],
    'std_db5_cD1': [],
    'var_db5_cD1': [],
    'sb_energy_db5_cD1': [],
    'skewness_db5_cD1': [],
    '5th_percentile_db5_cD1': [],
    '25th_percentile_db5_cD1': [],
    '75th_percentile_db5_cD1': [],
    '95th_percentile_db5_cD1': [],
    'rms_db5_cD1': [],
    'zcr_db5_cD1': [],
    'mcr_db5_cD1': [],
    'entropy_db5_cD1': [],

    'mean_db8_cA7': [],
    'median_db8_cA7': [],
    'std_db8_cA7': [],
    'var_db8_cA7': [],
    'sb_energy_db8_cA7': [],
    'skewness_db8_cA7': [],
    '5th_percentile_db8_cA7': [],
    '25th_percentile_db8_cA7': [],
    '75th_percentile_db8_cA7': [],
    '95th_percentile_db8_cA7': [],
    'rms_db8_cA7': [],
    'zcr_db8_cA7': [],
    'mcr_db8_cA7': [],
    'entropy_db8_cA7': [],

    'mean_db8_cD7': [],
    'median_db8_cD7': [],
    'std_db8_cD7': [],
    'var_db8_cD7': [],
    'sb_energy_db8_cD7': [],
    'skewness_db8_cD7': [],
    '5th_percentile_db8_cD7': [],
    '25th_percentile_db8_cD7': [],
    '75th_percentile_db8_cD7': [],
    '95th_percentile_db8_cD7': [],
    'rms_db8_cD7': [],
    'zcr_db8_cD7': [],
    'mcr_db8_cD7': [],
    'entropy_db8_cD7': [],

    'mean_db8_cD6': [],
    'median_db8_cD6': [],
    'std_db8_cD6': [],
    'var_db8_cD6': [],
    'sb_energy_db8_cD6': [],
    'skewness_db8_cD6': [],
    '5th_percentile_db8_cD6': [],
    '25th_percentile_db8_cD6': [],
    '75th_percentile_db8_cD6': [],
    '95th_percentile_db8_cD6': [],
    'rms_db8_cD6': [],
    'zcr_db8_cD6': [],
    'mcr_db8_cD6': [],
    'entropy_db8_cD6': [],

    'mean_db8_cD5': [],
    'median_db8_cD5': [],
    'std_db8_cD5': [],
    'var_db8_cD5': [],
    'sb_energy_db8_cD5': [],
    'skewness_db8_cD5': [],
    '5th_percentile_db8_cD5': [],
    '25th_percentile_db8_cD5': [],
    '75th_percentile_db8_cD5': [],
    '95th_percentile_db8_cD5': [],
    'rms_db8_cD5': [],
    'zcr_db8_cD5': [],
    'mcr_db8_cD5': [],
    'entropy_db8_cD5': [],

    'mean_db8_cD4': [],
    'median_db8_cD4': [],
    'std_db8_cD4': [],
    'var_db8_cD4': [],
    'sb_energy_db8_cD4': [],
    'skewness_db8_cD4': [],
    '5th_percentile_db8_cD4': [],
    '25th_percentile_db8_cD4': [],
    '75th_percentile_db8_cD4': [],
    '95th_percentile_db8_cD4': [],
    'rms_db8_cD4': [],
    'zcr_db8_cD4': [],
    'mcr_db8_cD4': [],
    'entropy_db8_cD4': [],

    'mean_db8_cD3': [],
    'median_db8_cD3': [],
    'std_db8_cD3': [],
    'var_db8_cD3': [],
    'sb_energy_db8_cD3': [],
    'skewness_db8_cD3': [],
    '5th_percentile_db8_cD3': [],
    '25th_percentile_db8_cD3': [],
    '75th_percentile_db8_cD3': [],
    '95th_percentile_db8_cD3': [],
    'rms_db8_cD3': [],
    'zcr_db8_cD3': [],
    'mcr_db8_cD3': [],
    'entropy_db8_cD3': [],

    'mean_db8_cD2': [],
    'median_db8_cD2': [],
    'std_db8_cD2': [],
    'var_db8_cD2': [],
    'sb_energy_db8_cD2': [],
    'skewness_db8_cD2': [],
    '5th_percentile_db8_cD2': [],
    '25th_percentile_db8_cD2': [],
    '75th_percentile_db8_cD2': [],
    '95th_percentile_db8_cD2': [],
    'rms_db8_cD2': [],
    'zcr_db8_cD2': [],
    'mcr_db8_cD2': [],
    'entropy_db8_cD2': [],

    'mean_db8_cD1': [],
    'median_db8_cD1': [],
    'std_db8_cD1': [],
    'var_db8_cD1': [],
    'sb_energy_db8_cD1': [],
    'skewness_db8_cD1': [],
    '5th_percentile_db8_cD1': [],
    '25th_percentile_db8_cD1': [],
    '75th_percentile_db8_cD1': [],
    '95th_percentile_db8_cD1': [],
    'rms_db8_cD1': [],
    'zcr_db8_cD1': [],
    'mcr_db8_cD1': [],
    'entropy_db8_cD1': [],

})


def feedback(file, genre_label):
    """
    feedback() is a extract function from extract_audio_features(), taking
    as parameter a file object and genre label. Prints the status of the 
    feature extraction process.
    """
    if type(file) == str:
        print('appended features extracted from ' + file + ' with genre: ' + genre_label)
    else:
        print('appended features extracted from ' + str(file.name) + ' with genre: ' + genre_label)


def extract_audio_features(dataframe, file, genre_label, data_source):
    """
    This function takes a dataframe, an audio file (check librosa for acceptable formats),
    genre label, and data source. It extract features from the audio and returns the
    dataframe with the new row appended.

    Timbral, rhythmic, and wavelet features are extracted excluding log-mel and mel-spectogram.

    Parameters:
    dataframe (pandas.Dataframe): Dataframe to be updated with new row.
    file (File or str): an audio file or file path.
    genre_label (str): audio genre label
    data_source (str): fma or gtzan
    """

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
        '5th_percentile_db4_cA4': stats_db4_cA4['5th_percentile'],
        '25th_percentile_db4_cA4': stats_db4_cA4['25th_percentile'],
        '75th_percentile_db4_cA4': stats_db4_cA4['75th_percentile'],
        '95th_percentile_db4_cA4': stats_db4_cA4['95th_percentile'],
        'rms_db4_cA4': stats_db4_cA4['rms'],
        'zcr_db4_cA4': stats_db4_cA4['zcr'],
        'mcr_db4_cA4': stats_db4_cA4['mcr'],
        'entropy_db4_cA4': stats_db4_cA4['entropy'],

        'mean_db4_cD4': stats_db4_cD4['mean'],
        'median_db4_cD4': stats_db4_cD4['median'],
        'std_db4_cD4': stats_db4_cD4['std'],
        'var_db4_cD4': stats_db4_cD4['var'],
        'sb_energy_db4_cD4': stats_db4_cD4['sb_energy'],
        'skewness_db4_cD4': stats_db4_cD4['skewness'],
        '5th_percentile_db4_cD4': stats_db4_cD4['5th_percentile'],
        '25th_percentile_db4_cD4': stats_db4_cD4['25th_percentile'],
        '75th_percentile_db4_cD4': stats_db4_cD4['75th_percentile'],
        '95th_percentile_db4_cD4': stats_db4_cD4['95th_percentile'],
        'rms_db4_cD4': stats_db4_cD4['rms'],
        'zcr_db4_cD4': stats_db4_cD4['zcr'],
        'mcr_db4_cD4': stats_db4_cD4['mcr'],
        'entropy_db4_cD4': stats_db4_cD4['entropy'],

        'mean_db4_cD3': stats_db4_cD3['mean'],
        'median_db4_cD3': stats_db4_cD3['median'],
        'std_db4_cD3': stats_db4_cD3['std'],
        'var_db4_cD3': stats_db4_cD3['var'],
        'sb_energy_db4_cD3': stats_db4_cD3['sb_energy'],
        'skewness_db4_cD3': stats_db4_cD3['skewness'],
        '5th_percentile_db4_cD3': stats_db4_cD3['5th_percentile'],
        '25th_percentile_db4_cD3': stats_db4_cD3['25th_percentile'],
        '75th_percentile_db4_cD3': stats_db4_cD3['75th_percentile'],
        '95th_percentile_db4_cD3': stats_db4_cD3['95th_percentile'],
        'rms_db4_cD3': stats_db4_cD3['rms'],
        'zcr_db4_cD3': stats_db4_cD3['zcr'],
        'mcr_db4_cD3': stats_db4_cD3['mcr'],
        'entropy_db4_cD3': stats_db4_cD3['entropy'],

        'mean_db4_cD2': stats_db4_cD2['mean'],
        'median_db4_cD2': stats_db4_cD2['median'],
        'std_db4_cD2': stats_db4_cD2['std'],
        'var_db4_cD2': stats_db4_cD2['var'],
        'sb_energy_db4_cD2': stats_db4_cD2['sb_energy'],
        'skewness_db4_cD2': stats_db4_cD2['skewness'],
        '5th_percentile_db4_cD2': stats_db4_cD2['5th_percentile'],
        '25th_percentile_db4_cD2': stats_db4_cD2['25th_percentile'],
        '75th_percentile_db4_cD2': stats_db4_cD2['75th_percentile'],
        '95th_percentile_db4_cD2': stats_db4_cD2['95th_percentile'],
        'rms_db4_cD2': stats_db4_cD2['rms'],
        'zcr_db4_cD2': stats_db4_cD2['zcr'],
        'mcr_db4_cD2': stats_db4_cD2['mcr'],
        'entropy_db4_cD2': stats_db4_cD2['entropy'],

        'mean_db4_cD1': stats_db4_cD1['mean'],
        'median_db4_cD1': stats_db4_cD1['median'],
        'std_db4_cD1': stats_db4_cD1['std'],
        'var_db4_cD1': stats_db4_cD1['var'],
        'sb_energy_db4_cD1': stats_db4_cD1['sb_energy'],
        'skewness_db4_cD1': stats_db4_cD1['skewness'],
        '5th_percentile_db4_cD1': stats_db4_cD1['5th_percentile'],
        '25th_percentile_db4_cD1': stats_db4_cD1['25th_percentile'],
        '75th_percentile_db4_cD1': stats_db4_cD1['75th_percentile'],
        '95th_percentile_db4_cD1': stats_db4_cD1['95th_percentile'],
        'rms_db4_cD1': stats_db4_cD1['rms'],
        'zcr_db4_cD1': stats_db4_cD1['zcr'],
        'mcr_db4_cD1': stats_db4_cD1['mcr'],
        'entropy_db4_cD1': stats_db4_cD1['entropy'],

        'mean_db5_cA4': stats_db5_cA4['mean'],
        'median_db5_cA4': stats_db5_cA4['median'],
        'std_db5_cA4': stats_db5_cA4['std'],
        'var_db5_cA4': stats_db5_cA4['var'],
        'sb_energy_db5_cA4': stats_db5_cA4['sb_energy'],
        'skewness_db5_cA4': stats_db5_cA4['skewness'],
        '5th_percentile_db5_cA4': stats_db5_cA4['5th_percentile'],
        '25th_percentile_db5_cA4': stats_db5_cA4['25th_percentile'],
        '75th_percentile_db5_cA4': stats_db5_cA4['75th_percentile'],
        '95th_percentile_db5_cA4': stats_db5_cA4['95th_percentile'],
        'rms_db5_cA4': stats_db5_cA4['rms'],
        'zcr_db5_cA4': stats_db5_cA4['zcr'],
        'mcr_db5_cA4': stats_db5_cA4['mcr'],
        'entropy_db5_cA4': stats_db5_cA4['entropy'],

        'mean_db5_cD4': stats_db5_cD4['mean'],
        'median_db5_cD4': stats_db5_cD4['median'],
        'std_db5_cD4': stats_db5_cD4['std'],
        'var_db5_cD4': stats_db5_cD4['var'],
        'sb_energy_db5_cD4': stats_db5_cD4['sb_energy'],
        'skewness_db5_cD4': stats_db5_cD4['skewness'],
        '5th_percentile_db5_cD4': stats_db5_cD4['5th_percentile'],
        '25th_percentile_db5_cD4': stats_db5_cD4['25th_percentile'],
        '75th_percentile_db5_cD4': stats_db5_cD4['75th_percentile'],
        '95th_percentile_db5_cD4': stats_db5_cD4['95th_percentile'],
        'rms_db5_cD4': stats_db5_cD4['rms'],
        'zcr_db5_cD4': stats_db5_cD4['zcr'],
        'mcr_db5_cD4': stats_db5_cD4['mcr'],
        'entropy_db5_cD4': stats_db5_cD4['entropy'],

        'mean_db5_cD3': stats_db5_cD3['mean'],
        'median_db5_cD3': stats_db5_cD3['median'],
        'std_db5_cD3': stats_db5_cD3['std'],
        'var_db5_cD3': stats_db5_cD3['var'],
        'sb_energy_db5_cD3': stats_db5_cD3['sb_energy'],
        'skewness_db5_cD3': stats_db5_cD3['skewness'],
        '5th_percentile_db5_cD3': stats_db5_cD3['5th_percentile'],
        '25th_percentile_db5_cD3': stats_db5_cD3['25th_percentile'],
        '75th_percentile_db5_cD3': stats_db5_cD3['75th_percentile'],
        '95th_percentile_db5_cD3': stats_db5_cD3['95th_percentile'],
        'rms_db5_cD3': stats_db5_cD3['rms'],
        'zcr_db5_cD3': stats_db5_cD3['zcr'],
        'mcr_db5_cD3': stats_db5_cD3['mcr'],
        'entropy_db5_cD3': stats_db5_cD3['entropy'],

        'mean_db5_cD2': stats_db5_cD2['mean'],
        'median_db5_cD2': stats_db5_cD2['median'],
        'std_db5_cD2': stats_db5_cD2['std'],
        'var_db5_cD2': stats_db5_cD2['var'],
        'sb_energy_db5_cD2': stats_db5_cD2['sb_energy'],
        'skewness_db5_cD2': stats_db5_cD2['skewness'],
        '5th_percentile_db5_cD2': stats_db5_cD2['5th_percentile'],
        '25th_percentile_db5_cD2': stats_db5_cD2['25th_percentile'],
        '75th_percentile_db5_cD2': stats_db5_cD2['75th_percentile'],
        '95th_percentile_db5_cD2': stats_db5_cD2['95th_percentile'],
        'rms_db5_cD2': stats_db5_cD2['rms'],
        'zcr_db5_cD2': stats_db5_cD2['zcr'],
        'mcr_db5_cD2': stats_db5_cD2['mcr'],
        'entropy_db5_cD2': stats_db5_cD2['entropy'],

        'mean_db5_cD1': stats_db5_cD1['mean'],
        'median_db5_cD1': stats_db5_cD1['median'],
        'std_db5_cD1': stats_db5_cD1['std'],
        'var_db5_cD1': stats_db5_cD1['var'],
        'sb_energy_db5_cD1': stats_db5_cD1['sb_energy'],
        'skewness_db5_cD1': stats_db5_cD1['skewness'],
        '5th_percentile_db5_cD1': stats_db5_cD1['5th_percentile'],
        '25th_percentile_db5_cD1': stats_db5_cD1['25th_percentile'],
        '75th_percentile_db5_cD1': stats_db5_cD1['75th_percentile'],
        '95th_percentile_db5_cD1': stats_db5_cD1['95th_percentile'],
        'rms_db5_cD1': stats_db5_cD1['rms'],
        'zcr_db5_cD1': stats_db5_cD1['zcr'],
        'mcr_db5_cD1': stats_db5_cD1['mcr'],
        'entropy_db5_cD1': stats_db5_cD1['entropy'],

        'mean_db8_cA7': stats_db8_cA7['mean'],
        'median_db8_cA7': stats_db8_cA7['median'],
        'std_db8_cA7': stats_db8_cA7['std'],
        'var_db8_cA7': stats_db8_cA7['var'],
        'sb_energy_db8_cA7': stats_db8_cA7['sb_energy'],
        'skewness_db8_cA7': stats_db8_cA7['skewness'],
        '5th_percentile_db8_cA7': stats_db8_cA7['5th_percentile'],
        '25th_percentile_db8_cA7': stats_db8_cA7['25th_percentile'],
        '75th_percentile_db8_cA7': stats_db8_cA7['75th_percentile'],
        '95th_percentile_db8_cA7': stats_db8_cA7['95th_percentile'],
        'rms_db8_cA7': stats_db8_cA7['rms'],
        'zcr_db8_cA7': stats_db8_cA7['zcr'],
        'mcr_db8_cA7': stats_db8_cA7['mcr'],
        'entropy_db8_cA7': stats_db8_cA7['entropy'],

        'mean_db8_cD7': stats_db8_cD7['mean'],
        'median_db8_cD7': stats_db8_cD7['median'],
        'std_db8_cD7': stats_db8_cD7['std'],
        'var_db8_cD7': stats_db8_cD7['var'],
        'sb_energy_db8_cD7': stats_db8_cD7['sb_energy'],
        'skewness_db8_cD7': stats_db8_cD7['skewness'],
        '5th_percentile_db8_cD7': stats_db8_cD7['5th_percentile'],
        '25th_percentile_db8_cD7': stats_db8_cD7['25th_percentile'],
        '75th_percentile_db8_cD7': stats_db8_cD7['75th_percentile'],
        '95th_percentile_db8_cD7': stats_db8_cD7['95th_percentile'],
        'rms_db8_cD7': stats_db8_cD7['rms'],
        'zcr_db8_cD7': stats_db8_cD7['zcr'],
        'mcr_db8_cD7': stats_db8_cD7['mcr'],
        'entropy_db8_cD7': stats_db8_cD7['entropy'],

        'mean_db8_cD6': stats_db8_cD6['mean'],
        'median_db8_cD6': stats_db8_cD6['median'],
        'std_db8_cD6': stats_db8_cD6['std'],
        'var_db8_cD6': stats_db8_cD6['var'],
        'sb_energy_db8_cD6': stats_db8_cD6['sb_energy'],
        'skewness_db8_cD6': stats_db8_cD6['skewness'],
        '5th_percentile_db8_cD6': stats_db8_cD6['5th_percentile'],
        '25th_percentile_db8_cD6': stats_db8_cD6['25th_percentile'],
        '75th_percentile_db8_cD6': stats_db8_cD6['75th_percentile'],
        '95th_percentile_db8_cD6': stats_db8_cD6['95th_percentile'],
        'rms_db8_cD6': stats_db8_cD6['rms'],
        'zcr_db8_cD6': stats_db8_cD6['zcr'],
        'mcr_db8_cD6': stats_db8_cD6['mcr'],
        'entropy_db8_cD6': stats_db8_cD6['entropy'],

        'mean_db8_cD5': stats_db8_cD5['mean'],
        'median_db8_cD5': stats_db8_cD5['median'],
        'std_db8_cD5': stats_db8_cD5['std'],
        'var_db8_cD5': stats_db8_cD5['var'],
        'sb_energy_db8_cD5': stats_db8_cD5['sb_energy'],
        'skewness_db8_cD5': stats_db8_cD5['skewness'],
        '5th_percentile_db8_cD5': stats_db8_cD5['5th_percentile'],
        '25th_percentile_db8_cD5': stats_db8_cD5['25th_percentile'],
        '75th_percentile_db8_cD5': stats_db8_cD5['75th_percentile'],
        '95th_percentile_db8_cD5': stats_db8_cD5['95th_percentile'],
        'rms_db8_cD5': stats_db8_cD5['rms'],
        'zcr_db8_cD5': stats_db8_cD5['zcr'],
        'mcr_db8_cD5': stats_db8_cD5['mcr'],
        'entropy_db8_cD5': stats_db8_cD5['entropy'],

        'mean_db8_cD4': stats_db8_cD4['mean'],
        'median_db8_cD4': stats_db8_cD4['median'],
        'std_db8_cD4': stats_db8_cD4['std'],
        'var_db8_cD4': stats_db8_cD4['var'],
        'sb_energy_db8_cD4': stats_db8_cD4['sb_energy'],
        'skewness_db8_cD4': stats_db8_cD4['skewness'],
        '5th_percentile_db8_cD4': stats_db8_cD4['5th_percentile'],
        '25th_percentile_db8_cD4': stats_db8_cD4['25th_percentile'],
        '75th_percentile_db8_cD4': stats_db8_cD4['75th_percentile'],
        '95th_percentile_db8_cD4': stats_db8_cD4['95th_percentile'],
        'rms_db8_cD4': stats_db8_cD4['rms'],
        'zcr_db8_cD4': stats_db8_cD4['zcr'],
        'mcr_db8_cD4': stats_db8_cD4['mcr'],
        'entropy_db8_cD4': stats_db8_cD4['entropy'],

        'mean_db8_cD3': stats_db8_cD3['mean'],
        'median_db8_cD3': stats_db8_cD3['median'],
        'std_db8_cD3': stats_db8_cD3['std'],
        'var_db8_cD3': stats_db8_cD3['var'],
        'sb_energy_db8_cD3': stats_db8_cD3['sb_energy'],
        'skewness_db8_cD3': stats_db8_cD3['skewness'],
        '5th_percentile_db8_cD3': stats_db8_cD3['5th_percentile'],
        '25th_percentile_db8_cD3': stats_db8_cD3['25th_percentile'],
        '75th_percentile_db8_cD3': stats_db8_cD3['75th_percentile'],
        '95th_percentile_db8_cD3': stats_db8_cD3['95th_percentile'],
        'rms_db8_cD3': stats_db8_cD3['rms'],
        'zcr_db8_cD3': stats_db8_cD3['zcr'],
        'mcr_db8_cD3': stats_db8_cD3['mcr'],
        'entropy_db8_cD3': stats_db8_cD3['entropy'],

        'mean_db8_cD2': stats_db8_cD2['mean'],
        'median_db8_cD2': stats_db8_cD2['median'],
        'std_db8_cD2': stats_db8_cD2['std'],
        'var_db8_cD2': stats_db8_cD2['var'],
        'sb_energy_db8_cD2': stats_db8_cD2['sb_energy'],
        'skewness_db8_cD2': stats_db8_cD2['skewness'],
        '5th_percentile_db8_cD2': stats_db8_cD2['5th_percentile'],
        '25th_percentile_db8_cD2': stats_db8_cD2['25th_percentile'],
        '75th_percentile_db8_cD2': stats_db8_cD2['75th_percentile'],
        '95th_percentile_db8_cD2': stats_db8_cD2['95th_percentile'],
        'rms_db8_cD2': stats_db8_cD2['rms'],
        'zcr_db8_cD2': stats_db8_cD2['zcr'],
        'mcr_db8_cD2': stats_db8_cD2['mcr'],
        'entropy_db8_cD2': stats_db8_cD2['entropy'],

        'mean_db8_cD1': stats_db8_cD1['mean'],
        'median_db8_cD1': stats_db8_cD1['median'],
        'std_db8_cD1': stats_db8_cD1['std'],
        'var_db8_cD1': stats_db8_cD1['var'],
        'sb_energy_db8_cD1': stats_db8_cD1['sb_energy'],
        'skewness_db8_cD1': stats_db8_cD1['skewness'],
        '5th_percentile_db8_cD1': stats_db8_cD1['5th_percentile'],
        '25th_percentile_db8_cD1': stats_db8_cD1['25th_percentile'],
        '75th_percentile_db8_cD1': stats_db8_cD1['75th_percentile'],
        '95th_percentile_db8_cD1': stats_db8_cD1['95th_percentile'],
        'rms_db8_cD1': stats_db8_cD1['rms'],
        'zcr_db8_cD1': stats_db8_cD1['zcr'],
        'mcr_db8_cD1': stats_db8_cD1['mcr'],
        'entropy_db8_cD1': stats_db8_cD1['entropy'],
    }

    # append new row
    dataframe = dataframe.append(new_row, ignore_index=True)

    feedback(file, genre_label)

    return dataframe


def extract_features_make_prediction(filepath):
    """
    This function takes path to a .wav audio file as input.
    It extract features from the audio file, scales these
    features & make prediction with them.

    Parameter:
    filepath (str): path to a .wav audio file
    """
    
    # extract features
    features = extract_audio_features(dataframe, filepath, '', '')
    
    # drop some columns
    features = features.drop(['data_source', 'lpc_1'], axis=1)
    
    # filter wavelet features & collect labels
    wavelet_predictors = features.filter(regex=(r'.+_db[458]{1}_.+'))
    wavelet_predictors_labels = wavelet_predictors.columns.values
    
    # collect timbral & rhythmic features
    timbral_rhythmic_predictors = features.loc[:, features.columns.difference(
        numpy.append(wavelet_predictors_labels, 'genre_label'))]
    X = timbral_rhythmic_predictors

    pipeline_estimator_path = MOUNTED_DATASET_PATH + '/model/pipeline_estimator_3.pkl'
    model_path = MOUNTED_DATASET_PATH + '/model/cnn_model_3.h5'
    
    # load preprocessing pipeline and model instance
    pipeline_estimator = joblib.load(pipeline_estimator_path)
    model = load_model(model_path)
    
    # transform data & make predictions
    X = pipeline_estimator.transform(X)
    prediction = model.predict(X)
    
    # map predictions to genres
    map_prediction_to_genre = {}
    for i in range(3):
        map_prediction_to_genre[GENRES[i]] = prediction[0][i].item()

    return map_prediction_to_genre
