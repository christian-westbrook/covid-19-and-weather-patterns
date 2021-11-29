# File System
import os
import json
from pathlib import Path
from zipfile import ZipFile
import pickle

import numpy as np
import pandas as pd
from sympy.geometry import *


DATA_PATH = '../data/' # Point this constant to the location of your data archive files

EXPECTED_DATASETS = {'Colorado': [
    'county_total_population.Colorado.zip',
    'covid_county.Colorado.zip',
    'neon_2d_wind.Colorado.zip',
    'neon_barometric_pressure.Colorado.zip',
    'neon_single_asp_air_temperature.Colorado.zip'
]}
counties = {'Colorado': ['Boulder', 'Grand', 'Larimer', 'Logan', 'Weld', 'Yuma']}


def get_datasets(state='Colorado', recalc=False):
    # Returns dataframes in order: control, covidWind, covidPressure, covidTemperature
    if recalc:
        print("Building Datasets")
        extract(state)
        create_pickles(state)
    elif not all([os.path.exists(f'../data/control.{state}.pkl'), os.path.exists(f'../data/covidWind.{state}.pkl'), os.path.exists(f'../data/covidPressure.{state}.pkl'), os.path.exists(f'../data/covidTemperature.{state}.pkl')]):
        extract(state)
        print("One or more .pkl files does not exist. Rebuilding datasets.")
        create_pickles(state)

    return pd.read_pickle(f'../data/control.{state}.pkl'), pd.read_pickle(f'../data/covidWind.{state}.pkl'), pd.read_pickle(f'../data/covidPressure.{state}.pkl'), pd.read_pickle(f'../data/covidTemperature.{state}.pkl')



def extract(state):
    # For each listed dataset string in the EXPECTED_DATASETS constant
    for datasetName in EXPECTED_DATASETS[state]:
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


def create_pickles(state):
    print("Loading flattened datasets")
    flattenedCovidDataFrame = pd.json_normalize(json.load(open(Path(f'../data/covid_county.{state}/data.json'))))
    flattenedCovidGeometryFrame = pd.json_normalize(json.load(open(Path(f'../data/covid_county.{state}/linkedGeometry.json'))))

    flattenedWindDataFrame = pd.json_normalize(json.load(open(Path(f'../data/neon_2d_wind.{state}/data.json'))))
    flattenedWindGeometryFrame = pd.json_normalize(json.load(open(Path(f'../data/neon_2d_wind.{state}/linkedGeometry.json'))))

    flattenedPressureDataFrame = pd.json_normalize(json.load(open(Path(f'../data/neon_barometric_pressure.{state}/data.json'))))
    flattenedPressureGeometryFrame = pd.json_normalize(json.load(open(Path(f'../data/neon_barometric_pressure.{state}/linkedGeometry.json'))))

    flattenedTemperatureDataFrame = pd.json_normalize(json.load(open(Path(f'../data/neon_single_asp_air_temperature.{state}/data.json'))))
    flattenedTemperatureGeometryFrame = pd.json_normalize(json.load(open(Path(f'../data/neon_single_asp_air_temperature.{state}/linkedGeometry.json'))))

    flattenedPopulationDataFrame = pd.json_normalize(json.load(open(Path(f'../data/county_total_population.{state}/data.json'))))
    flattenedPopulationGeometryFrame = pd.json_normalize(json.load(open(Path(f'../data/county_total_population.{state}/linkedGeometry.json'))))

    print("Combining data geometry for COVID dataset")
    combinedCovidFrame = flattenedCovidDataFrame.set_index('GISJOIN').join(
        flattenedCovidGeometryFrame.set_index('GISJOIN'), lsuffix='_data', rsuffix='_geo')

    combinedCovidFrame = combinedCovidFrame[combinedCovidFrame.county.isin(counties[state])]
    print("Finding County Geometries")
    county_polygons = create_county_polygons(state, combinedCovidFrame)

    print("Adding County info to geometries")
    flattenedPressureGeometryFrame['county'] = flattenedPressureGeometryFrame.apply(
        lambda row: lookup_county_from_geometry(county_polygons, row['geometry.coordinates']), axis=1)
    flattenedWindGeometryFrame['county'] = flattenedWindGeometryFrame.apply(
        lambda row: lookup_county_from_geometry(county_polygons, row['geometry.coordinates']), axis=1)
    flattenedTemperatureGeometryFrame['county'] = flattenedTemperatureGeometryFrame.apply(
        lambda row: lookup_county_from_geometry(county_polygons, row['geometry.coordinates']), axis=1)

    print("Combining remaining datasets")
    combinedWindFrame = flattenedWindDataFrame.set_index('site').join(flattenedWindGeometryFrame.set_index('site'), lsuffix='_data', rsuffix='_geo')
    combinedPressureFrame = flattenedPressureDataFrame.set_index('site').join(flattenedPressureGeometryFrame.set_index('site'), lsuffix='_data', rsuffix='_geo')
    combinedTemperatureFrame = flattenedTemperatureDataFrame.set_index('site').join(flattenedTemperatureGeometryFrame.set_index('site'), lsuffix='_data', rsuffix='_geo')
    combinedPopulationFrame = flattenedPopulationDataFrame.set_index('GISJOIN').join(flattenedPopulationGeometryFrame.set_index('GISJOIN'), lsuffix='_data', rsuffix='_geo')

    combinedCovidFrame['date'] = pd.to_datetime(combinedCovidFrame['dateString']).dt.date
    combinedWindFrame['date'] = pd.to_datetime(combinedWindFrame['startDateTime']).dt.date
    combinedPressureFrame['date'] = pd.to_datetime(combinedPressureFrame['startDateTime']).dt.date
    combinedTemperatureFrame['date'] = pd.to_datetime(combinedTemperatureFrame['startDateTime']).dt.date

    combinedWindFrame['datetime'] = pd.to_datetime(combinedWindFrame['startDateTime']).dt.round("H")
    combinedPressureFrame['datetime'] = pd.to_datetime(combinedPressureFrame['startDateTime']).dt.round("H")
    combinedTemperatureFrame['datetime'] = pd.to_datetime(combinedTemperatureFrame['startDateTime']).dt.round("H")

    print("Saving data to .pkl files")
    combinedCovidFrame.to_pickle(f'../data/control.{state}.pkl')
    combinedPopulationFrame.to_pickle(f'../data/population.{state}.pkl')
    finalCovidWindFrame = pd.merge(combinedWindFrame, combinedCovidFrame, how='left', left_on=['county', 'date'],
                                   right_on=['county', 'date'])
    finalCovidWindFrame.to_pickle(f'../data/covidWind.{state}.pkl')

    finalCovidPressureFrame = pd.merge(combinedPressureFrame, combinedCovidFrame, how='left',
                                       left_on=['county', 'date'], right_on=['county', 'date'])
    finalCovidPressureFrame.to_pickle(f'../data/covidPressure.{state}.pkl')

    finalCovidTemperatureFrame = pd.merge(combinedWindFrame, combinedCovidFrame, how='left', left_on=['county', 'date'],
                                          right_on=['county', 'date'])
    finalCovidTemperatureFrame.to_pickle(f'../data/covidTemperature.{state}.pkl')



def lookup_county_from_geometry(county_polygons, geometry):
    point = Point(np.asarray(geometry))
    for county, shape in county_polygons:
        if shape.encloses_point(point):
            print(f"found {county} for point")
            return county


def create_county_polygons(state, combinedCovidFrame):
    if not os.path.exists(DATA_PATH + f'/county_objects.{state}.pickle'):
        county_polygons = create_county_polygons(combinedCovidFrame, state)
        with open(DATA_PATH + f'/county_objects.{state}.pickle', 'wb') as handle:
            pickle.dump(county_polygons, handle)
    else:
        with open(DATA_PATH + f'/county_objects.{state}.pickle', 'rb') as handle:
            county_polygons = pickle.load(handle)
    return county_polygons



def _create_county_polygons(combinedCovidFrame):
    county_polygons = []
    df = combinedCovidFrame.groupby('county').first().reindex(columns=['coordJsonString', 'geometry.type', "geometry.coordinates"])
    print(df)
    df['coordJsonString'] = df["geometry.coordinates"].apply(json.dumps)
    for index, row in df[['coordJsonString', 'geometry.type']].iterrows():
        shape_data = json.loads(row['coordJsonString'])
        county = index
        if row['geometry.type'] == 'MultiPolygon':
            for p in shape_data:
                s = np.asarray(p)[0]
                shape = Polygon(*s)
                county_polygons.append((county, shape))
        elif row['geometry.type'] == 'Polygon':
            s = np.asarray(shape_data)[0]
            shape = Polygon(*s)
            county_polygons.append((county, shape))
        else:
            print(f"Row had geometry type of {row['geometry.type']}, row is {row}")
    return county_polygons