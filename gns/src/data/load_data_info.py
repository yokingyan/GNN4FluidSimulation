"""
    This File is to load the information about dataset and configs
"""
import json
import os


def read_metadata(data_path, file_name='metadata.json'):
    """Read json file and return a dict."""
    with open(os.path.join(data_path, file_name), 'rt') as fp:
        return json.loads(fp.read())
