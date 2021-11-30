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


    data = tensorflow.data.Dataset.from_tensor_slices((train_features.values, train_labels.values)).random().repeat().batch(64)
    print(data)
    #for row in data.take(5):
    #    print(row)
    linear_model.fit(dataset, epochs=5,steps_per_epoch=10)
    #print("Error")
    #print(linear_model.evaluate(tensorflow.data.Dataset.from_tensors((test_features.values, test_labels.values))))
    #print("Actual - 10")
    #print(test_labels[:10])
    #print("Predicted - 10")
    #print(linear_model.predict(test_features[:10]))


def dnn(train_features, train_labels, test_features, test_labels):
    if distributed_strategy is not None:
        with distributed_strategy.scope():
            normalizer = keras.layers.Normalization(axis=-1)
            normalizer.adapt(np.array(train_features))
            dnn_model = keras.Sequential([
                normalizer,
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(1)
            ])
            dnn_model.compile(loss='mean_absolute_error', optimizer=tensorflow.keras.optimizers.Adam(0.001))
    else:
        normalizer = keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))
        dnn_model = keras.Sequential([
            normalizer,
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1)
        ])
        dnn_model.compile(loss='mean_absolute_error', optimizer=tensorflow.keras.optimizers.Adam(0.001))

    dnn_model.fit(
        train_features.to_numpy(),
        train_labels.to_numpy(),
        epochs=100)
    print("Error")
    print(dnn_model.evaluate(test_features.to_numpy(), test_labels.to_numpy(), verbose=0))
    print("Actual - 10")
    print(test_labels[:10])
    print("Predicted - 10")
    print(dnn_model.predict(test_features[:10]))


def build_and_test_models(dataframe):
    train = dataframe.sample(frac=0.8)
    test = dataframe.drop(train.index)
    train_features = train.copy()
    test_features = test.copy()
    train_labels = train_features.pop('newCaseCount')
    test_labels = test_features.pop('newCaseCount')
    print("Running Linear Regression for dataset")
    print('train_features')
    print(train_features)

    print('train_labels')
    print(train_labels)

    print('test_features')
    print(test_features)

    print('test_labels')
    print(test_labels)

    #linear_regression(train_features, train_labels, test_features, test_labels)
    #print("Running DNN for dataset")
    dnn(train_features, train_labels, test_features, test_labels)


def run_control(controlFrame):
    controlFrame['date'] = pd.to_datetime(controlFrame['dateString']).dt.strftime("%Y%m%d").astype(int)
    reducedControlFrame = controlFrame[
        ['date', 'county', 'newCaseCount', 'totalCaseCount', 'totalDeathCount', 'newDeathCount']]
    counties = {'Boulder': 1, 'Grand': 2, 'Larimer': 3, 'Logan': 4, 'Weld': 5, 'Yuma': 6}
    reducedControlFrame['county_index'] = reducedControlFrame.apply(lambda row: counties[row['county']], axis=1)
    reducedControlFrame = reducedControlFrame[['date', 'county_index', 'newCaseCount']]
    reducedControlFrame.set_index(['date', 'county_index'], inplace=True, drop=False)

    build_and_test_models(reducedControlFrame)


def prepare_distributed_training(index):
    global distributed_strategy
    with open('../distributed.yaml') as stream:
        distributed_conf = yaml.safe_load(stream)
    tf_config = {
        'cluster': {
#            'chief': distributed_conf['chief'],
            'worker': distributed_conf['worker']
#            'ps': distributed_conf['ps'],
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
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.distributed:
        prepare_distributed_training(args.index)

    control, covidWind, covidPressure, covidTemperature = data_processing.get_datasets()
    run_control(control)

