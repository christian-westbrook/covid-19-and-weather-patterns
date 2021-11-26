
import os
import sys
import json
import argparse
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd


# ----steps to run-----

# create virtual env
# activate virtual env
# install packages if needed (tensorflow, pandas, numpy)

# (saturn) python3 covid_dataset.py --index 0
# (neptune) python3 covid_dataset.py --index 1
# (mercury) python3 covid_dataset.py --index 2
# (mars) python3 covid_dataset.py --index 3


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)
# if '.' not in sys.path:
#   sys.path.insert(0, '.')

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, required=True)
args = parser.parse_args()

tf_config = {
    'cluster': {
        'worker': ['saturn:3087', 'neptune:3087', 'mercury:3087', 'mars:3087']
    },
    'task': {'type': 'worker', 'index': 0}
}

tf_config['task']['index'] = args.index

os.environ['TF_CONFIG'] = json.dumps(tf_config)



DATA_PATH = './data/' # Point this constant to the location of your data archive files

EXPECTED_DATASETS = [
    'county_total_population.Colorado.zip',
    'covid_county.Colorado.zip'
]

# For each listed dataset string in the EXPECTED_DATASETS constant
for datasetName in EXPECTED_DATASETS:
    try:
        # Open the given archive file
        with ZipFile(DATA_PATH + datasetName, 'r') as currentZip:
            # Build the target directory path for extracted data
            datasetNameTokens = datasetName.split('.')
            datasetNameTokens.remove('zip')
            targetDirectory = DATA_PATH + '.'.join(datasetNameTokens)
            
            # If the target directory doesn't exist, create it
            if not os.path.exists(targetDirectory):
                Path(targetDirectory).mkdir()
            
            # Extract all data from the archive file to the target directory
            currentZip.extractall(targetDirectory)
    except FileNotFoundError:
        print("Unable to open " + datasetName + " at path " + DATA_PATH + datasetName)
        
covidDataFrame = pd.io.json.read_json('./data/covid_county.Colorado/data.json')
# drop state
covidDataFrame.drop(labels=['state'], axis=1,inplace = True)
# map county to unique numerical number
covidDataFrame['county'] = pd.Categorical(covidDataFrame['county'], categories=covidDataFrame['county'].unique()).codes
# map date sting to numerical value, probably do this different elsewhere
covidDataFrame['dateString'] = pd.Categorical(covidDataFrame['dateString'], categories=covidDataFrame['dateString'].unique()).codes
# these fields will be dealt with elsewhere
covidDataFrame.drop(labels=['_id', 'GISJOIN', 'epoch_time'], axis=1,inplace = True)
# predict new case count so move to front
life_expectancy = covidDataFrame['newCaseCount']
covidDataFrame.drop(labels=['newCaseCount'], axis=1,inplace = True)
covidDataFrame.insert(0, 'newCaseCount', life_expectancy)
covidDataFrame.sample(5)



def _partition(X, T, train_fraction):
    n_samples = X.shape[0]
    rows = np.arange(n_samples)
    np.random.shuffle(rows)
    
    ntrain = round(n_samples * train_fraction)
    
    Xtrain = X[rows[:ntrain], :]
    Ttrain = T[rows[:ntrain], :]
    Xtest = X[rows[ntrain:], :]
    Ttest = T[rows[ntrain:], :]
    
    return Xtrain, Ttrain, Xtest, Ttest

def _rmse(T, Y):
    return np.sqrt(np.mean((T - Y)**2))

def covid_dataset():
    X = covidDataFrame.to_numpy()[:,1:]
    T = covidDataFrame.to_numpy()[:,:1]
    
    Xtrain, Ttrain, Xtest, Ttest = _partition(X, T, .8)
    return Xtrain, Ttrain, Xtest, Ttest
    
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(10)
    ])
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    model = build_model()
    Xtrain, Ttrain, Xtest, Ttest = covid_dataset()

    model.fit(Xtrain, Ttrain, epochs=500)
    
Y = model(Xtest)
print('RMSE:', _rmse(Y, Ttest))
