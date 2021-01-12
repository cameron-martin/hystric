import os
from pathlib import Path

def get_data_dir():
    data_dir = os.environ.get('TFDS_DATA_DIR')
    if data_dir is None:
        data_dir = '~/tensorflow_datasets'
    return Path(data_dir)