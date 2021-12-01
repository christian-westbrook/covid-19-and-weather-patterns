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
from matplotlib import pyplot

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


def dnn(train_features, train_labels, val_features, val_labels, learning_rate, decay_steps, decay_rate, epochs, patience):

    # Hyperparameters
    learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate, 
        decay_steps=decay_steps, 
        decay_rate=decay_rate
    )

    custom_adam = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

    if distributed_strategy is not None:
        with distributed_strategy.scope():
            normalizer = keras.layers.Normalization(axis=-1)
            normalizer.adapt(np.array(train_features))
            dnn_model = keras.Sequential([
                normalizer,
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(1)
            ])

            dnn_model.compile(loss='mae', optimizer=custom_adam, metrics=['mae', 'mse'])
    else:
        normalizer = keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))
        dnn_model = keras.Sequential([
            normalizer,
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1)
        ])

        dnn_model.compile(loss='mae', optimizer=custom_adam, metrics=['mae', 'mse'])

    history = dnn_model.fit(
        train_features.to_numpy(),
        train_labels.to_numpy(),
        batch_size=2048,
        epochs=epochs,
        validation_data=(val_features.to_numpy(), val_labels.to_numpy()),
        verbose=0,
        callbacks=[early_stop]
    )

    return dnn_model, history

    # Visually compare a few samples
    # print("Actual - 10")
    # print(test_labels[:10])
    # print("Predicted - 10")
    # print(dnn_model.predict(test_features[:10]))


def build_and_test_models(dataframe):
    # Training testing validating split
    train = dataframe.sample(frac=0.6)
    dataframe = dataframe.drop(train.index)

    val = dataframe.sample(frac=0.5)
    test = dataframe.drop(val.index)

    train_features = train.copy()
    val_features = val.copy()
    test_features = test.copy()
    train_labels = train_features.pop('newCaseCount')
    val_labels = val_features.pop('newCaseCount')
    test_labels = test_features.pop('newCaseCount')

    # Hyperparameters
    learning_rate = 0.001 # Adam initial learning rate
    decay_steps = 35 # Steps before learning rate decay
    decay_rate = 0.98 # Rate at which learning rate decays
    epochs = 1000 # Number of complete passes across the training set
    patience = 35 # Number of steps with no progress we'll tolerate before early stopping

    print("Running DNN for dataset")
    model, history = dnn(train_features, train_labels, val_features, val_labels, learning_rate, decay_steps, decay_rate, epochs, patience)

    pyplot.plot(history.history['loss'], label='Train Loss')
    pyplot.plot(history.history['val_loss'], label = 'Validation Loss')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend(loc='lower right')

    loss, mae, mse = model.evaluate(test_features.to_numpy(), test_labels.to_numpy(), verbose=1)

    return loss, mae, mse


def run_temperature(df, pop):
    reducedPopFrame = pop[['county', '2010_total_population']]
    df['date'] = pd.to_datetime(df['dateString']).dt.strftime("%Y%m%d").astype(int)
    df = pd.merge(df, reducedPopFrame, how='left', left_on=['county'], right_on=['county'])
    df['county'] = pd.Categorical(df['county'], categories=df['county'].unique()).codes
    df = df[['date', 'county', 'newCaseCount', 'newDeathCount', 'totalCaseCount', 'totalDeathCount', 'tempSingleMean', 'tempSingleMaximum', 'tempSingleMinimum', 'tempSingleVariance']]
    return build_and_test_models(df)


def run_pressure(df, pop):
    reducedPopFrame = pop[['county', '2010_total_population']]
    df['date'] = pd.to_datetime(df['dateString']).dt.strftime("%Y%m%d").astype(int)
    df = pd.merge(df, reducedPopFrame, how='left', left_on=['county'], right_on=['county'])
    df['county'] = pd.Categorical(df['county'], categories=df['county'].unique()).codes
    df = df[['date', 'county', 'newCaseCount', 'newDeathCount', 'totalCaseCount', 'totalDeathCount', 'corPres', 'staPresMean', 'staPresMaximum', 'staPresMinimum']]
    return build_and_test_models(df)


def run_wind(df, pop):
    reducedPopFrame = pop[['county', '2010_total_population']]
    df['date'] = pd.to_datetime(df['dateString']).dt.strftime("%Y%m%d").astype(int)
    df = pd.merge(df, reducedPopFrame, how='left', left_on=['county'], right_on=['county'])
    df['county'] = pd.Categorical(df['county'], categories=df['county'].unique()).codes
    df = df[['date', 'county', 'newCaseCount', 'newDeathCount', 'totalCaseCount', 'totalDeathCount', 'windSpeedMinimum', 'windSpeedMean', 'windSpeedMaximum']]
    return build_and_test_models(df)


def run_control(controlFrame, pop):
    reducedPopFrame = pop[['county', '2010_total_population']]
    controlFrame['date'] = pd.to_datetime(controlFrame['dateString']).dt.strftime("%Y%m%d").astype(int)
    reducedControlFrame = controlFrame[
        ['date', 'county', 'newCaseCount', 'totalCaseCount', 'totalDeathCount', 'newDeathCount']]
    reducedControlFrame = pd.merge(reducedControlFrame, reducedPopFrame, how='left', left_on=['county'], right_on=['county'])
    counties = {'Boulder': 1, 'Grand': 2, 'Larimer': 3, 'Logan': 4, 'Weld': 5, 'Yuma': 6}
    reducedControlFrame['county_index'] = reducedControlFrame.apply(lambda row: counties[row['county']], axis=1)
    reducedControlFrame = reducedControlFrame[['date', 'county_index', 'newCaseCount', 'totalCaseCount', 'totalDeathCount', 'newDeathCount', '2010_total_population']]
    reducedControlFrame.set_index(['date', 'county_index'], inplace=True, drop=False)
    print(reducedControlFrame.columns)
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

    control, covidWind, covidPressure, covidTemperature, pop = data_processing.get_datasets()
    error_values = {}
    if args.model == 'all':
        error_values['control'] = run_control(control, pop)
        error_values['temperature'] = run_temperature(covidTemperature, pop)
        error_values['pressure'] = run_pressure(covidPressure, pop)
        error_values['wind'] = run_wind(covidWind, pop)
    elif args.model == 'control':
        error_values['control'] = run_control(control, pop)
    elif args.model == 'temperature':
        error_values['temperature'] = run_temperature(covidTemperature, pop)
    elif args.model == 'pressure':
        error_values['pressure'] = run_pressure(covidPressure, pop)
    elif args.model == 'wind':
        error_values['wind'] = run_wind(covidWind, pop)

    print("mean_absolute_error")
    for dsname, error_value in error_values.items():
        print(f"{dsname}\t\t{error_value}")
