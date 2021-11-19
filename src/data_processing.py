import json
import pandas as pd


def load_flattened_datasets(data_path, field_labels_path, linked_geometry_path, join_on_key=None):
    data_df = pd.json_normalize(json.load(open(data_path)))
    print(data_df.info())
    field_labels_df = pd.json_normalize(json.load(open(field_labels_path)))
    linked_geometry_df = pd.json_normalize(json.load(open(linked_geometry_path)))
    joined = data_df.set_index(join_on_key).join(linked_geometry_df.set_index(join_on_key), lsuffix='_data', rsuffix='_geo')
    return joined

    # data_df = pd.read_json(data_path)
    # field_labels_df = pd.read_json(field_labels_path)
    # linked_geometry_df = pd.read_json(linked_geometry_path)


