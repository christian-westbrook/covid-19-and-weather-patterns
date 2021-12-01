import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow
from tensorflow import keras
import argparse
import yaml
import data_processing
import json

distributed_strategy = None
"""
To run this distributed, add workers to distributed.yaml. For example:
worker:
  - "earth:11889"
  - "jupiter:11891"
  - "saturn:11891"
  - "venus:11981"
  
These workers must be started with the index in this file.

[earth] python run.py --distribute --index 0
[jupiter] python run.py --distribute --index 1
[saturn] python run.py --distribute --index 2
[venus] python run.py --distribute --index 3

"""



def linear_regression(train_features, train_labels, test_features, test_labels):
    if distributed_strategy is not None:
        with distributed_strategy.scope():
            normalizer = keras.layers.Normalization(axis=-1)
            normalizer.adapt(np.array(train_features))
            linear_model = keras.Sequential([
                normalizer,
                keras.layers.Dense(units=1)
            ])
            linear_model.compile(
                optimizer=tensorflow.optimizers.Adam(learning_rate=0.1),
                loss='mean_absolute_error')
    else:
        normalizer = keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))
        linear_model = keras.Sequential([
            normalizer,
            keras.layers.Dense(units=1)
        ])
    linear_model.compile(
        optimizer=tensorflow.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')

    linear_model.fit(
        train_features,
        train_labels,
        epochs=20,
        # Calculate validation results on 20% of the training data.
        validation_split=0.2)
    print("Error")
    print(linear_model.evaluate(test_features, test_labels, verbose=0))
    print("Actual - 10")
    print(test_labels[:10])
    print("Predicted - 10")
    print(linear_model.predict(test_features[:10]))


def dnn(train_features, train_labels, test_features, test_labels):
    if distributed_strategy is not None:
        with distributed_strategy.scope():
            normalizer = keras.layers.Normalization(axis=-1)
            normalizer.adapt(np.array(train_features))
            dnn_model = keras.Sequential([
                normalizer,
                keras.layers.Dense(100, activation='relu'),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dense(8, activation='relu'),
                keras.layers.Dense(1)
            ])
            dnn_model.compile(loss='mean_absolute_error', optimizer=tensorflow.keras.optimizers.Adam(0.001))
    else:
        normalizer = keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))
        dnn_model = keras.Sequential([
            normalizer,
            keras.layers.Dense(100, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(1)
        ])

    dnn_model.compile(loss='mean_absolute_error', optimizer=tensorflow.keras.optimizers.Adam(0.001))
    dnn_model.fit(
        train_features.to_numpy(),
        train_labels.to_numpy(),
        epochs=75)
    return dnn_model.evaluate(test_features.to_numpy(), test_labels.to_numpy(), verbose=0)

    # Visually compare a few samples
    # print("Actual - 10")
    # print(test_labels[:10])
    # print("Predicted - 10")
    # print(dnn_model.predict(test_features[:10]))


def build_and_test_models(dataframe):
    train = dataframe.sample(frac=0.8)
    test = dataframe.drop(train.index)
    train_features = train.copy()
    test_features = test.copy()
    train_labels = train_features.pop('newCaseCount')
    test_labels = test_features.pop('newCaseCount')
    print("Running DNN for dataset")
    return dnn(train_features, train_labels, test_features, test_labels)


def run_temperature(df):
    df['date'] = pd.to_datetime(df['dateString']).dt.strftime("%Y%m%d").astype(int)
    df['county'] = pd.Categorical(df['county'], categories=df['county'].unique()).codes
    df = df[['date', 'county', 'newCaseCount', 'newDeathCount', 'tempSingleMean', 'tempSingleMaximum', 'tempSingleMinimum', 'tempSingleVariance']]
    return build_and_test_models(df)


def run_pressure(df):
    df['date'] = pd.to_datetime(df['dateString']).dt.strftime("%Y%m%d").astype(int)
    df['county'] = pd.Categorical(df['county'], categories=df['county'].unique()).codes
    df = df[['date', 'county', 'newCaseCount', 'newDeathCount', 'corPres', 'staPresMean', 'staPresMaximum', 'staPresMinimum']]
    return build_and_test_models(df)


def run_wind(df):
    df['date'] = pd.to_datetime(df['dateString']).dt.strftime("%Y%m%d").astype(int)
    df['county'] = pd.Categorical(df['county'], categories=df['county'].unique()).codes
    df = df[['date', 'county', 'newCaseCount', 'newDeathCount', 'windSpeedMinimum', 'windSpeedMean', 'windSpeedMaximum']]
    return build_and_test_models(df)


def run_control(controlFrame):
    controlFrame['date'] = pd.to_datetime(controlFrame['dateString']).dt.strftime("%Y%m%d").astype(int)
    reducedControlFrame = controlFrame[
        ['date', 'county', 'newCaseCount', 'totalCaseCount', 'totalDeathCount', 'newDeathCount']]
    counties = {'Boulder': 1, 'Grand': 2, 'Larimer': 3, 'Logan': 4, 'Weld': 5, 'Yuma': 6}
    reducedControlFrame['county_index'] = reducedControlFrame.apply(lambda row: counties[row['county']], axis=1)
    reducedControlFrame = reducedControlFrame[['date', 'county_index', 'newCaseCount', 'newDeathCount']]
    reducedControlFrame.set_index(['date', 'county_index'], inplace=True, drop=False)
    return build_and_test_models(reducedControlFrame)


def prepare_distributed_training(index):
    global distributed_strategy
    with open('../distributed.yaml') as stream:
        distributed_conf = yaml.safe_load(stream)
    tf_config = {
        'cluster': {
            'worker': distributed_conf['worker']
        },
        'task': {'type': 'worker', 'index': index}
    }
    print(tf_config)
    os.environ['TF_CONFIG'] = json.dumps(tf_config)
    cluster_resolver = tensorflow.distribute.cluster_resolver.TFConfigClusterResolver()
    distributed_strategy = tensorflow.distribute.MultiWorkerMirroredStrategy(cluster_resolver)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--model', choices=['all', 'control', 'temperature', 'pressure', 'wind'], default='all')
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.distributed:
        prepare_distributed_training(args.index)

    control, covidWind, covidPressure, covidTemperature = data_processing.get_datasets()
    error_values = {}
    if args.model == 'all':
        error_values['control'] = run_control(control)
        error_values['temperature'] = run_temperature(covidTemperature)
        error_values['pressure'] = run_pressure(covidPressure)
        error_values['wind'] = run_wind(covidWind)
    elif args.model == 'control':
        error_values['control'] = run_control(control)
    elif args.model == 'temperature':
        error_values['temperature'] = run_temperature(covidTemperature)
    elif args.model == 'pressure':
        error_values['pressure'] = run_pressure(covidPressure)
    elif args.model == 'wind':
        error_values['wind'] = run_wind(covidWind)

    print("mean_absolute_error")
    for dsname, error_value in error_values.items():
        print(f"{dsname}\t\t{error_value}")

